package main

import (
	"bytes"
	"database/sql"
	"embed"
	"encoding/json"
	"encoding/xml"
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

type NameCount struct {
	Name  string
	Count int
}

// RSS 2.0 feed types. These exist only to be marshaled by encoding/xml --
// string fields are plain chardata so the encoder handles escaping (article
// titles routinely contain curly quotes and ampersands).
type rssFeed struct {
	XMLName xml.Name   `xml:"rss"`
	Version string     `xml:"version,attr"`
	Atom    string     `xml:"xmlns:atom,attr"`
	Channel rssChannel `xml:"channel"`
}

type rssChannel struct {
	Title         string      `xml:"title"`
	Link          string      `xml:"link"`
	Description   string      `xml:"description"`
	Language      string      `xml:"language"`
	LastBuildDate string      `xml:"lastBuildDate,omitempty"`
	AtomLink      rssAtomLink `xml:"atom:link"`
	Items         []rssItem   `xml:"item"`
}

type rssAtomLink struct {
	Href string `xml:"href,attr"`
	Rel  string `xml:"rel,attr"`
	Type string `xml:"type,attr"`
}

type rssItem struct {
	Title       string   `xml:"title"`
	Link        string   `xml:"link"`
	GUID        rssGUID  `xml:"guid"`
	PubDate     string   `xml:"pubDate,omitempty"`
	Description string   `xml:"description"`
	Categories  []string `xml:"category"`
}

type rssGUID struct {
	Value       string `xml:",chardata"`
	IsPermaLink bool   `xml:"isPermaLink,attr"`
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
	TopicCounts  []NameCount
	SourceCounts []NameCount
	TotalCount   int
	DailySummary string
}

var db *sql.DB
var indexTmpl *template.Template
var statsTmpl *template.Template

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
		"domain":     fqdn,
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

	indexTmpl, err = template.New("").Funcs(funcMap).ParseFS(content,
		"templates/base.html", "templates/index.html", "templates/histogram.html")
	if err != nil {
		log.Fatalf("parse index templates: %v", err)
	}

	statsTmpl, err = template.New("").Funcs(funcMap).ParseFS(content,
		"templates/base.html", "templates/stats.html", "templates/histogram.html")
	if err != nil {
		log.Fatalf("parse stats templates: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/stats", handleStats)
	mux.HandleFunc("/rss", handleRSS)
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

	recentDates := lastNDates(5)

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
	if err := indexTmpl.ExecuteTemplate(w, "base.html", data); err != nil {
		log.Println("template:", err)
	}
}

const statsDays = 10

// handleStats serves a stand-alone page summarizing article activity over
// the last statsDays days: the daily histogram plus per-topic and
// per-source (FQDN) counts.
func handleStats(w http.ResponseWriter, r *http.Request) {
	topics, err := getTopics()
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getTopics:", err)
		return
	}

	total, err := getTotalCount()
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getTotalCount:", err)
		return
	}

	dailyCounts, err := getDailyCounts(statsDays)
	if err != nil {
		log.Println("getDailyCounts:", err)
		dailyCounts = nil
	}

	topicCounts, err := getTopicCounts(statsDays)
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getTopicCounts:", err)
		return
	}

	sourceCounts, err := getSourceCounts(statsDays)
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getSourceCounts:", err)
		return
	}

	data := PageData{
		Topics:       topics,
		TotalCount:   total,
		RecentDates:  lastNDates(5),
		DailyCounts:  dailyCounts,
		TopicCounts:  topicCounts,
		SourceCounts: sourceCounts,
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := statsTmpl.ExecuteTemplate(w, "base.html", data); err != nil {
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

const (
	feedDefaultLimit = 50
	feedMaxLimit     = 500
)

// baseURL reconstructs this site's absolute origin from the request. Nginx
// sets Host and X-Forwarded-Proto, so this is correct in production and in
// local dev without a hardcoded domain or extra configuration.
func baseURL(r *http.Request) string {
	scheme := r.Header.Get("X-Forwarded-Proto")
	if scheme == "" {
		scheme = "http"
	}
	return scheme + "://" + r.Host
}

// rssPubDate converts a stored fetched_at into the RFC1123Z format RSS
// requires. A handful of legacy rows hold a bare date rather than a full
// timestamp; those fall back to midnight, and anything unparseable returns ""
// so the pubDate element is omitted entirely rather than emitted malformed.
func rssPubDate(fetchedAt string) string {
	if t, err := time.Parse(time.RFC3339Nano, fetchedAt); err == nil {
		return t.Format(time.RFC1123Z)
	}
	if t, err := time.Parse("2006-01-02", fetchedAt); err == nil {
		return t.Format(time.RFC1123Z)
	}
	return ""
}

// handleRSS serves the article list as an RSS 2.0 feed, honoring the same
// topic/date filters as the listing page plus a ?limit= override.
func handleRSS(w http.ResponseWriter, r *http.Request) {
	topic := r.URL.Query().Get("topic")
	dateFilter := r.URL.Query().Get("date")

	limit := feedDefaultLimit
	if n, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil {
		limit = n
	}
	if limit < 1 {
		limit = 1
	}
	if limit > feedMaxLimit {
		limit = feedMaxLimit
	}

	articles, err := getFeedArticles(topic, dateFilter, limit)
	if err != nil {
		http.Error(w, "db error", 500)
		log.Println("getFeedArticles:", err)
		return
	}

	base := baseURL(r)

	title := "Robot Overlord News"
	switch {
	case topic != "" && dateFilter != "":
		title += " — " + topic + ", " + dateFilter
	case topic != "":
		title += " — " + topic
	case dateFilter != "":
		title += " — " + dateFilter
	}

	items := make([]rssItem, 0, len(articles))
	for _, a := range articles {
		desc := a.Summary
		if a.PublishedOn != "" {
			// Original publication date belongs in the body, not pubDate --
			// see the ordering note below.
			desc = strings.TrimSpace(desc + "\n\nPublished: " + a.PublishedOn)
		}

		items = append(items, rssItem{
			Title: a.Title,
			Link:  a.URL,
			// url carries a unique index, so it's a stable permalink-style id.
			GUID: rssGUID{Value: a.URL, IsPermaLink: true},
			// pubDate is fetched_at, NOT published_on: readers re-sort and
			// de-duplicate by pubDate, so using the (often stale, often
			// missing) publication date here would silently discard the
			// save-order this feed is meant to syndicate.
			PubDate:     rssPubDate(a.FetchedAt),
			Description: desc,
			Categories:  a.Topics,
		})
	}

	lastBuild := ""
	if len(items) > 0 {
		lastBuild = items[0].PubDate
	}

	feed := rssFeed{
		Version: "2.0",
		Atom:    "http://www.w3.org/2005/Atom",
		Channel: rssChannel{
			Title:         title,
			Link:          base + "/",
			Description:   "Your new AI masters, summarized for your convenience.",
			Language:      "en-us",
			LastBuildDate: lastBuild,
			AtomLink: rssAtomLink{
				Href: base + r.URL.RequestURI(),
				Rel:  "self",
				Type: "application/rss+xml",
			},
			Items: items,
		},
	}

	w.Header().Set("Content-Type", "application/rss+xml; charset=utf-8")
	if _, err := w.Write([]byte(xml.Header)); err != nil {
		log.Println("handleRSS write header:", err)
		return
	}
	enc := xml.NewEncoder(w)
	enc.Indent("", "  ")
	if err := enc.Encode(feed); err != nil {
		log.Println("handleRSS encode:", err)
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

	nc := sortedNameCounts(counts)
	topics := make([]string, len(nc))
	for i, t := range nc {
		topics[i] = t.Name
	}
	return topics, nil
}

// sortedNameCounts converts a name->count map into a slice sorted by count
// desc, then name asc.
func sortedNameCounts(counts map[string]int) []NameCount {
	nc := make([]NameCount, 0, len(counts))
	for name, cnt := range counts {
		nc = append(nc, NameCount{name, cnt})
	}
	sort.Slice(nc, func(i, j int) bool {
		if nc[i].Count != nc[j].Count {
			return nc[i].Count > nc[j].Count
		}
		return nc[i].Name < nc[j].Name
	})
	return nc
}

// cutoffDate returns the earliest calendar date (inclusive) covered by a
// window of the last `days` days, ending today.
func cutoffDate(days int) string {
	return time.Now().AddDate(0, 0, -(days - 1)).Format("2006-01-02")
}

// getTopicCounts returns per-topic article counts for the last `days` days.
func getTopicCounts(days int) ([]NameCount, error) {
	rows, err := db.Query(`SELECT topic FROM news_results WHERE DATE(fetched_at) >= ?`, cutoffDate(days))
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
	return sortedNameCounts(counts), nil
}

// getSourceCounts returns per-source (FQDN) article counts for the last
// `days` days. URL parsing can't be done cleanly in SQL, so the tally
// happens in Go using fqdn().
func getSourceCounts(days int) ([]NameCount, error) {
	rows, err := db.Query(`SELECT url FROM news_results WHERE DATE(fetched_at) >= ?`, cutoffDate(days))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	counts := map[string]int{}
	for rows.Next() {
		var rawURL string
		if err := rows.Scan(&rawURL); err != nil {
			return nil, err
		}
		if d := fqdn(rawURL); d != "" {
			counts[d]++
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return sortedNameCounts(counts), nil
}

// getTotalCount returns the total number of articles, ignoring any filter.
func getTotalCount() (int, error) {
	var total int
	err := db.QueryRow(`SELECT COUNT(*) FROM news_results`).Scan(&total)
	return total, err
}

// fqdn extracts the hostname from a URL, stripped of a leading "www.".
func fqdn(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return ""
	}
	return strings.TrimPrefix(u.Hostname(), "www.")
}

// lastNDates returns the last n calendar dates (today first, oldest last)
// formatted as "2006-01-02".
func lastNDates(n int) []string {
	dates := make([]string, n)
	for i := 0; i < n; i++ {
		dates[i] = time.Now().AddDate(0, 0, -i).Format("2006-01-02")
	}
	return dates
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

// getFeedArticles returns the newest `limit` articles matching the topic/date
// filter, ordered by fetched_at desc. Rows with a blank url are excluded --
// an RSS item with an empty link and guid is invalid, and the data contains a
// few such rows.
func getFeedArticles(topic, date string, limit int) ([]Article, error) {
	clause, args := articleFilter(topic, date)
	where := ` WHERE COALESCE(url,'') != ''`
	if clause != "" {
		where += ` AND (` + clause + `)`
	}

	const cols = `SELECT id, topic, COALESCE(title,''), COALESCE(url,''), COALESCE(summary,''), fetched_at, COALESCE(published_on,'') FROM news_results`
	rows, err := db.Query(cols+where+` ORDER BY fetched_at DESC LIMIT ?`, append(append([]interface{}{}, args...), limit)...)
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

	rows, err := db.Query(
		`SELECT DATE(fetched_at) AS d, COUNT(*) FROM news_results WHERE DATE(fetched_at) >= ? GROUP BY d`,
		cutoffDate(days),
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

// topicColor deterministically maps a topic name to one of topicColors. It's a
// pure hash over a short string, so it's called fresh on every render rather
// than memoized -- a cache here would need locking to be safe across concurrent
// requests, and would cost more than it saves.
func topicColor(topic string) string {
	h := uint32(0)
	for _, c := range strings.ToLower(topic) {
		h = h*31 + uint32(c)
	}
	return topicColors[h%uint32(len(topicColors))]
}
