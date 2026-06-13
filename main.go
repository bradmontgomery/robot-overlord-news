package main

import (
	"database/sql"
	"embed"
	"html/template"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed templates/* static/*
var content embed.FS

const pageSize = 20

type Article struct {
	ID        int
	Topic     string
	Title     string
	URL       string
	Summary   string
	FetchedAt string
}

type PageData struct {
	Articles    []Article
	Topics      []string
	CurrentPage int
	TotalPages  int
	FilterTopic string
	TotalCount  int
}

var db *sql.DB
var tmpl *template.Template

func main() {
	dbPath := os.Getenv("NEWS_DB")
	if dbPath == "" {
		dbPath = "../langchain_simple_demo/news_data.db"
	}

	var err error
	db, err = sql.Open("sqlite", dbPath+"?_journal=WAL&mode=ro")
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	db.SetMaxOpenConns(1)

	if _, err = db.Exec("PRAGMA query_only = ON"); err != nil {
		log.Printf("warn: query_only pragma: %v", err)
	}

	funcMap := template.FuncMap{
		"add": func(a, b int) int { return a + b },
		"sub": func(a, b int) int { return a - b },
		"seq": func(n int) []int {
			s := make([]int, n)
			for i := range s {
				s[i] = i + 1
			}
			return s
		},
		"topicColor": topicColor,
		"truncate": func(s string, n int) string {
			if len(s) <= n {
				return s
			}
			return s[:n] + "…"
		},
		"formatTime": func(s string) string {
			t, err := time.Parse(time.RFC3339Nano, s)
			if err != nil {
				return s
			}
			return t.Format("Jan 2, 2006 3:04 PM")
		},
	}

	tmpl, err = template.New("").Funcs(funcMap).ParseFS(content, "templates/*.html")
	if err != nil {
		log.Fatalf("parse templates: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.Handle("/static/", http.FileServer(http.FS(content)))

	addr := ":8080"
	log.Printf("listening on http://localhost%s  db=%s", addr, dbPath)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	topic := r.URL.Query().Get("topic")
	pageStr := r.URL.Query().Get("page")
	page := 1
	if p, err := strconv.Atoi(pageStr); err == nil && p > 1 {
		page = p
	}

	topics, err := getTopics()
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getTopics:", err)
		return
	}

	total, articles, err := getArticles(topic, page)
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getArticles:", err)
		return
	}

	totalPages := int(math.Ceil(float64(total) / float64(pageSize)))
	if totalPages == 0 {
		totalPages = 1
	}

	data := PageData{
		Articles:    articles,
		Topics:      topics,
		CurrentPage: page,
		TotalPages:  totalPages,
		FilterTopic: topic,
		TotalCount:  total,
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.ExecuteTemplate(w, "base.html", data); err != nil {
		log.Println("template:", err)
	}
}

func getTopics() ([]string, error) {
	rows, err := db.Query(`SELECT topic, COUNT(*) as cnt FROM news_results GROUP BY topic ORDER BY cnt DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var topics []string
	for rows.Next() {
		var t string
		var cnt int
		if err := rows.Scan(&t, &cnt); err != nil {
			return nil, err
		}
		topics = append(topics, t)
	}
	return topics, rows.Err()
}

func getArticles(topic string, page int) (int, []Article, error) {
	offset := (page - 1) * pageSize

	var countRow *sql.Row
	var rows *sql.Rows
	var err error

	if topic != "" {
		countRow = db.QueryRow(`SELECT COUNT(*) FROM news_results WHERE topic = ?`, topic)
	} else {
		countRow = db.QueryRow(`SELECT COUNT(*) FROM news_results`)
	}

	var total int
	if err = countRow.Scan(&total); err != nil {
		return 0, nil, err
	}

	if topic != "" {
		rows, err = db.Query(
			`SELECT id, topic, COALESCE(title,''), COALESCE(url,''), COALESCE(summary,''), fetched_at
			 FROM news_results WHERE topic = ?
			 ORDER BY fetched_at DESC LIMIT ? OFFSET ?`,
			topic, pageSize, offset,
		)
	} else {
		rows, err = db.Query(
			`SELECT id, topic, COALESCE(title,''), COALESCE(url,''), COALESCE(summary,''), fetched_at
			 FROM news_results
			 ORDER BY fetched_at DESC LIMIT ? OFFSET ?`,
			pageSize, offset,
		)
	}
	if err != nil {
		return 0, nil, err
	}
	defer rows.Close()

	var articles []Article
	for rows.Next() {
		var a Article
		if err := rows.Scan(&a.ID, &a.Topic, &a.Title, &a.URL, &a.Summary, &a.FetchedAt); err != nil {
			return 0, nil, err
		}
		articles = append(articles, a)
	}
	return total, articles, rows.Err()
}

var topicColors = []string{
	"#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444",
	"#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
}

var colorCache = map[string]string{}

func topicColor(topic string) string {
	if c, ok := colorCache[topic]; ok {
		return c
	}
	h := 0
	for _, c := range strings.ToLower(topic) {
		h = h*31 + int(c)
	}
	if h < 0 {
		h = -h
	}
	color := topicColors[h%len(topicColors)]
	colorCache[topic] = color
	return color
}
