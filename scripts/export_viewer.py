"""Generate standalone HTML dataset viewer and CSV exports for IDE inspection."""
import json
import pandas as pd

def generate_inspector():
    df = pd.read_parquet("data/processed/zomato_cleaned.parquet")
    print(f"Total rows in parquet: {len(df)}")

    # 1. Curated diverse sample for instant responsive HTML browsing
    sample_df = pd.concat([
        df[df["city"] == "faridabad"].sort_values(by="votes", ascending=False).head(250),
        df[df["city"] == "delhi"].sort_values(by="votes", ascending=False).head(400),
        df[df["city"] == "gurgaon"].sort_values(by="votes", ascending=False).head(250),
        df[df["city"] == "noida"].sort_values(by="votes", ascending=False).head(150),
        df[df["city"].isin(["rohtak", "panipat", "hisar", "karnal", "ambala", "sonipat"])].sort_values(by="votes", ascending=False).head(150),
        df[df["city"] == "bangalore"].sort_values(by="votes", ascending=False).head(200),
    ]).drop_duplicates(subset=["name", "city", "locality"])

    records = []
    for _, row in sample_df.iterrows():
        cuisines_str = ", ".join([c.title() for c in row["cuisines"]]) if isinstance(row["cuisines"], (list, tuple)) else str(row["cuisines"])
        records.append({
            "name": str(row["name"]),
            "city": str(row["city"]).title(),
            "locality": str(row["locality"]).title(),
            "rating": float(row["rating"]),
            "votes": int(row["votes"]),
            "cost_for_two": int(row["cost_for_two"]),
            "budget_tier": str(row["budget_tier"]).upper(),
            "cuisines": cuisines_str,
            "address": str(row["address"]),
        })

    data_json = json.dumps(records, ensure_ascii=False)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zomato Restaurant Dataset Inspector</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --zomato-red: #E23744;
            --zomato-dark: #1C1C1C;
            --zomato-gray: #696969;
            --zomato-border: #E8E8E8;
            --zomato-bg: #F8F9FA;
            --rating-green: #24963F;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--zomato-bg);
            color: var(--zomato-dark);
            padding: 24px;
        }
        .header {
            background: #FFFFFF;
            padding: 24px 32px;
            border-radius: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            border: 1px solid var(--zomato-border);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .brand-logo {
            background: var(--zomato-red);
            color: white;
            font-size: 24px;
            font-weight: 800;
            font-style: italic;
            padding: 8px 16px;
            border-radius: 10px;
            letter-spacing: -0.5px;
        }
        .brand h1 {
            font-size: 22px;
            font-weight: 700;
            color: var(--zomato-dark);
        }
        .brand p {
            font-size: 13px;
            color: var(--zomato-gray);
            margin-top: 2px;
        }
        .stats-badge {
            background: #FFF5F5;
            color: var(--zomato-red);
            border: 1px solid #FCD4D4;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .filters {
            background: #FFFFFF;
            padding: 18px 24px;
            border-radius: 12px;
            border: 1px solid var(--zomato-border);
            margin-bottom: 20px;
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filters input, .filters select {
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #D0D5DD;
            font-size: 14px;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        .filters input:focus, .filters select:focus {
            border-color: var(--zomato-red);
            box-shadow: 0 0 0 3px rgba(226, 55, 68, 0.12);
        }
        .search-box {
            flex: 1;
            min-width: 260px;
        }
        .table-container {
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid var(--zomato-border);
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        thead {
            background: #FAFAFA;
            border-bottom: 2px solid var(--zomato-border);
        }
        th {
            padding: 14px 18px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #4A4A4A;
        }
        tbody tr {
            border-bottom: 1px solid #F0F0F0;
            transition: background 0.15s ease;
        }
        tbody tr:hover {
            background: #FFF9F9;
        }
        td {
            padding: 14px 18px;
            font-size: 14px;
            vertical-align: middle;
        }
        .restaurant-name {
            font-weight: 700;
            color: var(--zomato-dark);
            font-size: 14.5px;
        }
        .city-pill {
            background: #EDF2F7;
            color: #2D3748;
            font-size: 11.5px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .rating-badge {
            background: var(--rating-green);
            color: white;
            font-weight: 700;
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }
        .rating-badge.low {
            background: #E5A812;
        }
        .cost-col {
            font-weight: 600;
            color: #2B2B2B;
        }
        .budget-tag {
            font-size: 11px;
            font-weight: 700;
            padding: 3px 6px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }
        .budget-tag.LOW { background: #E6F4EA; color: #137333; }
        .budget-tag.MID { background: #FEF7E0; color: #B06000; }
        .budget-tag.HIGH { background: #FCE8E6; color: #C5221F; }
        .cuisines-text {
            color: #4A4A4A;
            font-size: 13px;
            max-width: 280px;
        }
        .pagination {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: #FAFAFA;
            border-top: 1px solid var(--zomato-border);
            font-size: 13px;
            color: var(--zomato-gray);
        }
        .btn {
            background: #FFFFFF;
            border: 1px solid #D0D5DD;
            padding: 8px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            color: var(--zomato-dark);
            transition: all 0.2s;
        }
        .btn:hover:not(:disabled) {
            border-color: var(--zomato-red);
            color: var(--zomato-red);
        }
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="brand">
            <div class="brand-logo">zomato</div>
            <div>
                <h1>Restaurant Dataset Inspector</h1>
                <p>Interactive table for project data: Faridabad, Delhi NCR, Haryana & Bangalore</p>
            </div>
        </div>
        <div class="stats-badge">
            📊 33,476 Total Canonical Records in Parquet
        </div>
    </div>

    <div class="filters">
        <input type="text" id="searchBox" class="search-box" placeholder="🔍 Search restaurant name, cuisine, or locality..." oninput="applyFilters()">
        <select id="cityFilter" onchange="applyFilters()">
            <option value="">All Cities (Delhi NCR, Haryana, Bangalore)</option>
            <option value="Faridabad">Faridabad</option>
            <option value="Delhi">Delhi</option>
            <option value="Gurgaon">Gurgaon</option>
            <option value="Noida">Noida</option>
            <option value="Bangalore">Bangalore</option>
        </select>
        <select id="minRatingFilter" onchange="applyFilters()">
            <option value="0">Min Rating: Any</option>
            <option value="3.5">★ 3.5 & Above</option>
            <option value="4.0">★ 4.0 & Above</option>
            <option value="4.2">★ 4.2 & Above</option>
        </select>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Restaurant</th>
                    <th>City</th>
                    <th>Locality</th>
                    <th>Rating</th>
                    <th>Votes</th>
                    <th>Cost for Two</th>
                    <th>Budget</th>
                    <th>Cuisines</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
        <div class="pagination">
            <span id="recordCount">Showing 0 of 0</span>
            <div style="display:flex; gap: 8px;">
                <button id="prevBtn" class="btn" onclick="prevPage()">← Previous</button>
                <button id="nextBtn" class="btn" onclick="nextPage()">Next →</button>
            </div>
        </div>
    </div>

    <script>
        const rawData = DATA_PLACEHOLDER;
        let filteredData = [...rawData];
        let currentPage = 1;
        const pageSize = 20;

        function applyFilters() {
            const search = document.getElementById('searchBox').value.toLowerCase().trim();
            const city = document.getElementById('cityFilter').value;
            const minRating = parseFloat(document.getElementById('minRatingFilter').value);

            filteredData = rawData.filter(item => {
                if (city && item.city !== city) return false;
                if (item.rating < minRating) return false;
                if (search) {
                    const matchName = item.name.toLowerCase().includes(search);
                    const matchCuisine = item.cuisines.toLowerCase().includes(search);
                    const matchLocality = item.locality.toLowerCase().includes(search);
                    if (!matchName && !matchCuisine && !matchLocality) return false;
                }
                return true;
            });

            currentPage = 1;
            renderTable();
        }

        function renderTable() {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            const start = (currentPage - 1) * pageSize;
            const end = Math.min(start + pageSize, filteredData.length);
            const pageData = filteredData.slice(start, end);

            pageData.forEach(item => {
                const tr = document.createElement('tr');
                const ratingClass = item.rating >= 4.0 ? 'rating-badge' : 'rating-badge low';
                tr.innerHTML = `
                    <td class="restaurant-name">${item.name}</td>
                    <td><span class="city-pill">${item.city}</span></td>
                    <td style="color:#696969;">${item.locality}</td>
                    <td><span class="${ratingClass}">★ ${item.rating > 0 ? item.rating.toFixed(1) : 'NEW'}</span></td>
                    <td style="color:#696969;">${item.votes.toLocaleString()}</td>
                    <td class="cost-col">₹${item.cost_for_two}</td>
                    <td><span class="budget-tag ${item.budget_tier}">${item.budget_tier}</span></td>
                    <td class="cuisines-text">${item.cuisines}</td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById('recordCount').innerText = `Showing ${filteredData.length > 0 ? start + 1 : 0} to ${end} of ${filteredData.length.toLocaleString()} restaurants`;
            document.getElementById('prevBtn').disabled = currentPage <= 1;
            document.getElementById('nextBtn').disabled = end >= filteredData.length;
        }

        function prevPage() {
            if (currentPage > 1) {
                currentPage--;
                renderTable();
            }
        }

        function nextPage() {
            if ((currentPage * pageSize) < filteredData.length) {
                currentPage++;
                renderTable();
            }
        }

        renderTable();
    </script>
</body>
</html>"""

    final_html = html_template.replace("DATA_PLACEHOLDER", data_json)
    with open("dataset_viewer.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("dataset_viewer.html written successfully!")

if __name__ == "__main__":
    generate_inspector()
