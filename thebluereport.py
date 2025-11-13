from fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
from typing import Any, List, Dict
import re

mcp = FastMCP("The Blue Report")

@mcp.tool
def get_top_stories() -> List[Dict[str, str]]:
    """Get all top stories from theblue.report front page.
    
    Returns a list of stories with their titles, URLs, domains, and engagement stats.
    """
    try:
        response = requests.get("https://theblue.report", timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        stories = []
        
        # Find all story sections - each story is in a <section> within <div class="group">
        sections = soup.select('main div.group section')
        
        for section in sections:
            # Get the title from h2 within div.content
            content_div = section.find('div', class_='content')
            if not content_div:
                continue
                
            heading = content_div.find('h2')
            if not heading:
                continue
                
            title_link = heading.find_parent('a') or heading.find('a')
            if not title_link:
                continue
                
            title = heading.get_text(strip=True)
            # Remove leading number and period (e.g., "1. " -> "")
            title = re.sub(r'^\d+\.\s*', '', title)
            url = title_link.get('href', '')
            
            # Get metadata from p.metadata
            meta_paragraph = section.find('p', class_='metadata')
            if meta_paragraph:
                # Extract domain from span.host
                host_span = meta_paragraph.find('span', class_='host')
                domain = host_span.get_text(strip=True) if host_span else ''
                
                # Extract stats from the last span (after bullet)
                all_spans = meta_paragraph.find_all('span')
                stats = ''
                for span in all_spans:
                    if 'host' not in span.get('class', []) and 'bullet' not in span.get('class', []):
                        stats = span.get_text(strip=True)
                        break
            else:
                domain = ''
                stats = ''
            
            stories.append({
                'title': title,
                'url': url,
                'domain': domain,
                'stats': stats
            })
        
        return stories
        
    except requests.RequestException as error:
        return [{'error': f'Failed to fetch stories: {str(error)}'}]
    except Exception as error:
        return [{'error': f'Error parsing stories: {str(error)}'}]

if __name__ == "__main__":
    mcp.run(transport="stdio")