package main

import (
	"bytes"
	"database/sql"
	"embed"
	"encoding/json"
	"html/template"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/yuin/goldmark"
	_ "modernc.org/sqlite"
)

//go:embed templates/* static/*
var content embed.FS

const pageSize = 20

type Article struct {
	ID          int
	Topics      []string
	Title       string
	URL         string
	Summary     string
	FetchedAt   string
	PublishedOn string
}

type DayCount struct {
	Date  string // 2006-01-02
	Label string // e.g. "Jun 22"
	Count int
	Pct   int // bar height as percent of the busiest day (0-100)
}

type PageData struct {
	Articles     []Article
	Topics       []string
	CurrentPage  int
	TotalPages   int
	FilterTopic  string
	FilterDate   string
	RecentDates  []string
	DailyCounts  []DayCount
	TotalCount   int
	DailySummary string
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
		"markdown": func(s string) template.HTML {
			var buf bytes.Buffer
			if err := goldmark.Convert([]byte(s), &buf); err != nil {
				return template.HTML(template.HTMLEscapeString(s))
			}
			return template.HTML(buf.String())
		},
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
		"domain": func(rawURL string) string {
			u, err := url.Parse(rawURL)
			if err != nil || u.Host == "" {
				return ""
			}
			host := u.Hostname()
			return strings.TrimPrefix(host, "www.")
		},
		"formatDate": func(dateStr string) string {
			t, err := time.Parse("2006-01-02", dateStr)
			if err != nil {
				return dateStr
			}
			today := time.Now().Format("2006-01-02")
			yesterday := time.Now().AddDate(0, 0, -1).Format("2006-01-02")
			switch dateStr {
			case today:
				return "Today (" + t.Format("Jan 2") + ")"
			case yesterday:
				return "Yesterday (" + t.Format("Jan 2") + ")"
			default:
				return t.Format("Jan 2")
			}
		},
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
		"toJSON": func(v interface{}) (string, error) {
			b, err := json.Marshal(v)
			if err != nil {
				return "", err
			}
			return string(b), nil
		},
	}

	tmpl, err = template.New("").Funcs(funcMap).ParseFS(content, "templates/*.html")
	if err != nil {
		log.Fatalf("parse templates: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/api/present", handlePresent)
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
	dateFilter := r.URL.Query().Get("date")
	pageStr := r.URL.Query().Get("page")
	page := 1
	if p, err := strconv.Atoi(pageStr); err == nil && p > 1 {
		page = p
	}

	recentDates := make([]string, 5)
	for i := 0; i < 5; i++ {
		recentDates[i] = time.Now().AddDate(0, 0, -i).Format("2006-01-02")
	}

	summaryDate := dateFilter
	if summaryDate == "" {
		summaryDate = time.Now().Format("2006-01-02")
	}
	dailySummary, err := getDailySummary(summaryDate)
	if err != nil {
		log.Println("getDailySummary:", err)
		dailySummary = ""
	}

	topics, err := getTopics()
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getTopics:", err)
		return
	}

	dailyCounts, err := getDailyCounts(10)
	if err != nil {
		log.Println("getDailyCounts:", err)
		dailyCounts = nil
	}

	total, articles, err := getArticles(topic, dateFilter, page)
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
		Articles:     articles,
		Topics:       topics,
		CurrentPage:  page,
		TotalPages:   totalPages,
		FilterTopic:  topic,
		FilterDate:   dateFilter,
		RecentDates:  recentDates,
		DailyCounts:  dailyCounts,
		TotalCount:   total,
		DailySummary: dailySummary,
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.ExecuteTemplate(w, "base.html", data); err != nil {
		log.Println("template:", err)
	}
}

// handlePresent serves the article list for presentation mode as JSON. It
// respects the topic/date filters, but only respects pagination if a page
// was explicitly selected in the query string — otherwise it returns every
// matching article so presentation mode isn't limited to the first page.
func handlePresent(w http.ResponseWriter, r *http.Request) {
	topic := r.URL.Query().Get("topic")
	dateFilter := r.URL.Query().Get("date")
	pageStr := r.URL.Query().Get("page")

	var articles []Article
	var err error

	if pageStr != "" {
		page := 1
		if p, perr := strconv.Atoi(pageStr); perr == nil && p > 1 {
			page = p
		}
		_, articles, err = getArticles(topic, dateFilter, page)
	} else {
		articles, err = getAllArticles(topic, dateFilter)
	}

	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("handlePresent:", err)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if err := json.NewEncoder(w).Encode(articles); err != nil {
		log.Println("handlePresent encode:", err)
	}
}

func getTopics() ([]string, error) {
	rows, err := db.Query(`SELECT topic FROM news_results`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	counts := map[string]int{}
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		for _, t := range splitTopics(raw) {
			counts[t]++
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	type topicCount struct {
		name  string
		count int
	}
	tc := make([]topicCount, 0, len(counts))
	for name, cnt := range counts {
		tc = append(tc, topicCount{name, cnt})
	}
	sort.Slice(tc, func(i, j int) bool {
		if tc[i].count != tc[j].count {
			return tc[i].count > tc[j].count
		}
		return tc[i].name < tc[j].name
	})

	topics := make([]string, len(tc))
	for i, t := range tc {
		topics[i] = t.name
	}
	return topics, nil
}

func splitTopics(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// articleFilter builds the WHERE clause (without the "WHERE" keyword) and its
// bind args for the optional topic/date filters shared by getArticles and
// getAllArticles. Returns an empty clause when neither filter is set.
func articleFilter(topic, date string) (string, []interface{}) {
	const topicFilter = `',' || topic || ',' LIKE '%,' || ? || ',%'`
	const dateFilter = `DATE(fetched_at) = ?`

	switch {
	case topic != "" && date != "":
		return topicFilter + ` AND ` + dateFilter, []interface{}{topic, date}
	case topic != "":
		return topicFilter, []interface{}{topic}
	case date != "":
		return dateFilter, []interface{}{date}
	default:
		return "", nil
	}
}

func scanArticles(rows *sql.Rows) ([]Article, error) {
	articles := []Article{}
	for rows.Next() {
		var a Article
		var rawTopic string
		if err := rows.Scan(&a.ID, &rawTopic, &a.Title, &a.URL, &a.Summary, &a.FetchedAt, &a.PublishedOn); err != nil {
			return nil, err
		}
		a.Topics = splitTopics(rawTopic)
		articles = append(articles, a)
	}
	return articles, rows.Err()
}

func getArticles(topic, date string, page int) (int, []Article, error) {
	offset := (page - 1) * pageSize

	clause, args := articleFilter(topic, date)
	where := ""
	if clause != "" {
		where = ` WHERE ` + clause
	}

	var total int
	if err := db.QueryRow(`SELECT COUNT(*) FROM news_results`+where, args...).Scan(&total); err != nil {
		return 0, nil, err
	}

	const cols = `SELECT id, topic, COALESCE(title,''), COALESCE(url,''), COALESCE(summary,''), fetched_at, COALESCE(published_on,'') FROM news_results`
	rows, err := db.Query(cols+where+` ORDER BY fetched_at DESC LIMIT ? OFFSET ?`, append(append([]interface{}{}, args...), pageSize, offset)...)
	if err != nil {
		return 0, nil, err
	}
	defer rows.Close()

	articles, err := scanArticles(rows)
	return total, articles, err
}

// getAllArticles returns every article matching the topic/date filter, with
// no pagination limit. Used by presentation mode when no page has been
// explicitly selected.
func getAllArticles(topic, date string) ([]Article, error) {
	clause, args := articleFilter(topic, date)
	where := ""
	if clause != "" {
		where = ` WHERE ` + clause
	}

	const cols = `SELECT id, topic, COALESCE(title,''), COALESCE(url,''), COALESCE(summary,''), fetched_at, COALESCE(published_on,'') FROM news_results`
	rows, err := db.Query(cols+where+` ORDER BY fetched_at DESC`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanArticles(rows)
}

// getDailyCounts returns article counts per day for the last `days` calendar
// days (inclusive of today), oldest first, with zero-fill for empty days. It
// is a fixed overview of all articles and ignores any active topic/date filter.
func getDailyCounts(days int) ([]DayCount, error) {
	now := time.Now()
	cutoff := now.AddDate(0, 0, -(days - 1)).Format("2006-01-02")

	rows, err := db.Query(
		`SELECT DATE(fetched_at) AS d, COUNT(*) FROM news_results WHERE DATE(fetched_at) >= ? GROUP BY d`,
		cutoff,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	counts := map[string]int{}
	for rows.Next() {
		var d string
		var c int
		if err := rows.Scan(&d, &c); err != nil {
			return nil, err
		}
		counts[d] = c
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	out := make([]DayCount, days)
	max := 0
	for i := 0; i < days; i++ {
		t := now.AddDate(0, 0, -(days - 1 - i))
		date := t.Format("2006-01-02")
		c := counts[date]
		if c > max {
			max = c
		}
		out[i] = DayCount{Date: date, Label: t.Format("Jan 2"), Count: c}
	}

	for i := range out {
		if max > 0 && out[i].Count > 0 {
			pct := out[i].Count * 100 / max
			if pct < 4 {
				pct = 4 // keep a sliver visible for low-but-nonzero days
			}
			out[i].Pct = pct
		}
	}
	return out, nil
}

func getDailySummary(date string) (string, error) {
	var s string
	err := db.QueryRow(`SELECT summary FROM news_summaries WHERE date = ?`, date).Scan(&s)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return s, err
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
