import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

api_url = "https://codeforces.com/api/problemset.problems" # Query the codeforces problems
response = requests.get(api_url)
data = response.json()

rated_problems = []

if data['status'] == 'OK':
    problems = data['result']['problems']
    for p in problems:
        if 'rating' in p:
            rated_problems.append({
                'contestId': p['contestId'],
                'index': p['index'],
                'name': p['name'],
                'rating': p['rating']
            })
            
print(f"Found {len(rated_problems)} rated problems. Scraping the first 5...")

# Function to scrape problem text
def scrape_problem_text(contest_id, index):
    url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    try:
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.content, 'html.parser')
        statement_div = soup.find('div', class_='problem-statement')
        if statement_div:
            return statement_div.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Failed to scrape {contest_id}{index}: {e}")
    return None

# Scrape the first 5 problems
for prob in rated_problems[:2000]:
    print(f"Scraping {prob['contestId']}{prob['index']}...")
    prob['text'] = scrape_problem_text(prob['contestId'], prob['index'])
    time.sleep(1.5) # Pause to avoid overloading the servers

# 4. Save to CSV
df = pd.DataFrame(rated_problems[:2000])
df.to_csv('problems.csv', index=False)
print("Saved to problems.csv!")