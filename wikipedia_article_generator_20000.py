import pandas as pd
import random
import time
import wikipediaapi
import os
import tempfile
import concurrent.futures
import re
import sys
from tqdm import tqdm

def collect_wikipedia_content(count=20000, min_words=200, max_workers=8, retry_attempts=3):
    """
    Collect human-written text from Wikipedia articles (enhanced for large-scale collection)
    
    Parameters:
    count (int): Target number of articles to collect
    min_words (int): Minimum number of words required for each text entry
    max_workers (int): Maximum number of concurrent workers for parallel processing
    retry_attempts (int): Number of retry attempts for API failures
    
    Returns:
    pandas.DataFrame: Dataframe containing collected content
    """
    
    # Extended topic list to get more diverse content
    topics = [
        # Technology and Computing
        "Artificial intelligence", "Machine learning", "Computer science", "Internet", 
        "World Wide Web", "Programming language", "Database", "Software engineering",
        "Computer network", "Cybersecurity", "Cloud computing", "Quantum computing",
        "Virtual reality", "Augmented reality", "Blockchain", "Cryptocurrency",
        "Robotics", "Automation", "Big data", "Internet of things",
        
        # Natural Sciences
        "Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Ecology",
        "Quantum mechanics", "Relativity", "Thermodynamics", "Genetics",
        "Evolution", "Molecular biology", "Neuroscience", "Particle physics",
        "Climate change", "Solar System", "Galaxy", "Cosmology", "Nuclear energy",
        "Photosynthesis", "Cellular respiration", "Periodic table", 
        
        # Mathematics
        "Mathematics", "Algebra", "Geometry", "Calculus", "Statistics", 
        "Probability", "Number theory", "Topology", "Game theory", "Cryptography",
        
        # Social Sciences
        "Psychology", "Sociology", "Anthropology", "Economics", "Political science",
        "Archaeology", "Linguistics", "Cognitive science", "Behavioral economics",
        "International relations", "Geopolitics", "Democracy", "Socialism", "Capitalism",
        "Globalization", "Urbanization", "Social movement", "Education",
        
        # History and Geography
        "History", "Ancient civilization", "Middle Ages", "Renaissance", "Industrial Revolution",
        "World War I", "World War II", "Cold War", "Colonialism", "Decolonization", 
        "Geography", "Continents", "Ocean", "Mountain", "Desert", "Climate zone", "Rainforest",
        "History of China", "History of India", "History of Europe", "History of Africa",
        "History of the Americas", "Ancient Egypt", "Roman Empire", "Byzantine Empire",
        "Ottoman Empire", "British Empire", "Soviet Union", "Ancient Greece",
        
        # Arts and Culture
        "Art", "Literature", "Music", "Film", "Theatre", "Dance", "Architecture",
        "Sculpture", "Painting", "Poetry", "Novel", "Classical music", "Jazz",
        "Rock music", "Hip hop", "Cinema", "Television", "Photography", "Fashion",
        "Cuisine", "Cultural heritage", "Folklore", "Mythology",
        
        # Philosophy and Religion
        "Philosophy", "Ethics", "Logic", "Metaphysics", "Existentialism", "Empiricism",
        "Rationalism", "Religion", "Christianity", "Islam", "Hinduism", "Buddhism",
        "Judaism", "Sikhism", "Taoism", "Atheism", "Agnosticism", "Spirituality",
        
        # Health and Medicine
        "Medicine", "Anatomy", "Physiology", "Disease", "Immunology", "Pharmacology",
        "Surgery", "Psychiatry", "Public health", "Epidemiology", "Nutrition",
        "Cancer", "Cardiovascular disease", "Infectious disease", "Neurology",
        "Vaccine", "Antibiotic", "Pandemic", "Mental health",
        
        # Sports and Recreation
        "Sport", "Olympic Games", "Football", "Basketball", "Tennis", "Golf",
        "Cricket", "Baseball", "Swimming", "Athletics", "Gymnastics", "Cycling",
        "Chess", "Video game", "Board game", "Recreation", "Tourism",
        
        # Business and Economics
        "Business", "Corporation", "Finance", "Investment", "Stock market", "Banking",
        "Entrepreneurship", "Marketing", "Advertising", "Management", "Supply chain",
        "International trade", "Economic growth", "Inflation", "Recession", "Taxation",
        
        # Law and Government
        "Law", "Constitution", "Human rights", "Criminal justice", "Civil law",
        "International law", "Government", "Election", "Diplomacy", "Military",
        "Public policy", "Legislation", "Judiciary", "Executive branch",
        
        # Environment and Sustainability
        "Environment", "Renewable energy", "Sustainability", "Pollution", "Conservation",
        "Biodiversity", "Ecosystem", "Deforestation", "Desertification", "Climate action",
        "Wildlife", "Endangered species", "Marine conservation", "Recycling",
        
        # Transportation
        "Transportation", "Automobile", "Railway", "Aviation", "Shipping",
        "Public transport", "Highway", "Bicycle", "Electric vehicle", "Space exploration"
    ]
    
    # Add more specific topics to increase diversity
    additional_topics = [
        # Countries and regions
        "United States", "China", "India", "Russia", "Brazil", "Japan", "Germany", "France",
        "United Kingdom", "Italy", "Canada", "Australia", "Mexico", "South Korea", "Indonesia",
        "Turkey", "Saudi Arabia", "Egypt", "South Africa", "Nigeria", "Kenya", "Argentina",
        
        # Cities
        "New York City", "Tokyo", "London", "Paris", "Berlin", "Moscow", "Beijing", "Shanghai",
        "Mumbai", "São Paulo", "Cairo", "Sydney", "Mexico City", "Los Angeles", "Chicago",
        
        # Historical periods
        "Ancient Rome", "Ancient China", "Mesopotamia", "Maya civilization", "Inca Empire",
        "Medieval Europe", "Enlightenment", "Victorian era", "Roaring Twenties", "Great Depression",
        
        # Technology companies
        "Apple Inc.", "Microsoft", "Google", "Amazon (company)", "Meta Platforms", "Tesla, Inc.",
        "Samsung", "Intel", "IBM", "Oracle Corporation",
        
        # Inventors and scientists
        "Albert Einstein", "Isaac Newton", "Marie Curie", "Nikola Tesla", "Charles Darwin",
        "Galileo Galilei", "Ada Lovelace", "Alan Turing", "Stephen Hawking", "Richard Feynman",
        
        # Writers and artists
        "William Shakespeare", "Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh",
        "Jane Austen", "Fyodor Dostoevsky", "Gabriel García Márquez", "Toni Morrison",
        
        # Miscellaneous
        "Dinosaur", "Space exploration", "Olympic Games", "World Cup", "Nobel Prize",
        "United Nations", "European Union", "NATO", "Human genome", "Renewable energy",
        "Artificial neural network", "Game design", "Film production", "Graphic design",
        "Architecture", "Interior design", "Fashion design", "Food science", "Culinary arts"
    ]
    
    topics.extend(additional_topics)
    
    print(f"Starting collection with {len(topics)} seed topics, targeting {count} articles...")
    
    # Initialize Wikipedia API with proper user agent
    user_agent = 'WikipediaTextCollection/1.0 (research-project@example.com)'
    wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent
    )
    
    # Function to fetch a page with retries
    def fetch_page_with_retry(topic, attempts=retry_attempts):
        for attempt in range(attempts):
            try:
                page = wiki.page(topic)
                if page.exists():
                    return page
                return None
            except Exception as e:
                if attempt < attempts - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    print(f"Failed to fetch '{topic}' after {attempts} attempts: {e}")
                    return None
    
    # Function to extract pages and their links
    def expand_topic(topic):
        try:
            page = fetch_page_with_retry(topic)
            if page is None or not page.exists():
                return []
            
            # Skip disambiguation and stub pages
            if "may refer to" in page.summary[:100] or len(page.text) < 10000:
                return []
            
            # Get linked pages
            links = list(page.links.values())
            random.shuffle(links)
            
            # Take up to 30 links, but filter out non-English titles and special pages
            filtered_links = []
            for link in links[:75]:  # Process more links to ensure we get enough valid ones
                # Skip non-article pages (categories, files, templates, etc.)
                if any(ns in link.title for ns in ['Category:', 'File:', 'Template:', 'Wikipedia:', 'Help:', 'Portal:']):
                    continue
                
                # Skip pages with non-English characters (simple heuristic)
                if not all(ord(c) < 128 for c in link.title):
                    continue
                
                filtered_links.append(link)
                if len(filtered_links) >= 30:
                    break
            
            return [page] + filtered_links
        except Exception as e:
            print(f"Error expanding topic '{topic}': {e}")
            return []
    
    # Collect pages using multiple workers
    all_pages = []
    
    # First expand all seed topics with parallel processing
    print("Expanding seed topics to collect articles...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_topic = {executor.submit(expand_topic, topic): topic for topic in topics}
        
        # Create progress bar
        with tqdm(total=len(topics), desc="Expanding topics") as pbar:
            for future in concurrent.futures.as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    pages = future.result()
                    all_pages.extend(pages)
                    pbar.set_postfix({"Pages": len(all_pages)})
                except Exception as e:
                    print(f"Error processing topic '{topic}': {e}")
                pbar.update(1)
    
    # Deduplicate pages by title
    unique_pages = {}
    for page in all_pages:
        if page.title not in unique_pages:
            unique_pages[page.title] = page
    
    all_pages = list(unique_pages.values())
    
    # Shuffle pages for variety
    random.shuffle(all_pages)
    
    print(f"Collected {len(all_pages)} unique pages from {len(topics)} seed topics")
    
    # Extract content from pages
    print(f"Extracting content that meets the {min_words} word minimum...")
    wiki_data = []
    processed_count = 0
    
    with tqdm(total=min(count, len(all_pages)), desc="Processing articles") as pbar:
        for page in all_pages:
            if processed_count >= count:
                break
                
            try:
                # Skip inappropriate pages
                if any(term in page.title.lower() for term in ['sexual', 'pornography', 'nude', 'adult', 'xxx']):
                    continue
                    
                # Extract sections for better sampling
                sections = []
                
                # Process page summary if it meets word count threshold
                summary_word_count = len(page.summary.split())
                if summary_word_count >= min_words:
                    sections.append({
                        'text': page.summary,
                        'word_count': summary_word_count,
                        'is_summary': True
                    })
                
                # Process page sections
                for section in page.sections:
                    # Skip very short sections
                    word_count = len(section.text.split())
                    if word_count >= min_words:
                        sections.append({
                            'text': section.text,
                            'word_count': word_count,
                            'is_summary': False
                        })
                
                # Add each qualifying section as a separate entry
                for section in sections:
                    # Clean the text (remove excessive whitespace, citations, etc.)
                    clean_text = re.sub(r'\[\d+\]|\s+', ' ', section['text']).strip()
                    
                    # Verify the clean text still meets the word count
                    clean_word_count = len(clean_text.split())
                    if clean_word_count >= min_words:
                        wiki_data.append({
                            'title': page.title,
                            'text': clean_text,
                            'word_count': clean_word_count,
                            'url': page.fullurl,
                            'source': 'wikipedia',
                            'label': 0,  # Human-written
                            'is_summary': section.get('is_summary', False)
                        })
                        
                        processed_count += 1
                        pbar.update(1)
                        pbar.set_postfix({"Collected": processed_count})
                        
                        if processed_count >= count:
                            break
                
                # Be nice to the API - small pause between processing pages
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError processing '{page.title}': {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(wiki_data)
    
    # Generate statistics
    word_counts = df['word_count']
    print(f"\nCollection complete!")
    print(f"Total entries collected: {len(df)}")
    print(f"Average word count: {word_counts.mean():.1f}")
    print(f"Min word count: {word_counts.min()}")
    print(f"Max word count: {word_counts.max()}")
    print(f"Median word count: {word_counts.median():.1f}")
    
    return df

def save_dataset(df, filename="wikipedia_articles_10k.csv", create_backup=True):
    """
    Save the DataFrame to a CSV file with robust error handling
    
    Parameters:
    df: DataFrame to save
    filename: Base output filename
    create_backup: Whether to create backup copies
    """
    # Define potential save locations
    save_locations = [
        filename,  # Current directory
        os.path.join(tempfile.gettempdir(), filename),  # Temp directory
        os.path.expanduser(f"~/{filename}"),  # Home directory
        f"./{filename}"  # Explicit current directory
    ]
    
    success = False
    saved_path = None
    
    # Try each location until successful
    for location in save_locations:
        try:
            df.to_csv(location, index=False, encoding='utf-8')
            print(f"Successfully saved to: {location}")
            success = True
            saved_path = location
            break
        except Exception as e:
            print(f"Could not save to {location}: {e}")
    
    # Create backup if requested and primary save was successful
    if success and create_backup and saved_path:
        try:
            backup_path = f"{saved_path}.backup"
            df.to_csv(backup_path, index=False, encoding='utf-8')
            print(f"Created backup at: {backup_path}")
        except Exception as e:
            print(f"Could not create backup: {e}")
    
    if not success:
        print("Could not save the file to disk. Showing sample instead:")
        print(df.head())
        
    return success, saved_path

def main():
    """Main function to run the Wikipedia content collector"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect Wikipedia articles for text analysis')
    parser.add_argument('--count', type=int, default=20000, help='Number of articles to collect')
    parser.add_argument('--min-words', type=int, default=200, help='Minimum words per article')
    parser.add_argument('--output', type=str, default='wikipedia_articles_10k.csv', help='Output filename')
    parser.add_argument('--workers', type=int, default=8, help='Number of concurrent workers')
    parser.add_argument('--retries', type=int, default=3, help='Number of API retry attempts')
    
    args = parser.parse_args()
    
    print(f"Starting Wikipedia article collection:")
    print(f"Target count: {args.count}")
    print(f"Minimum words: {args.min_words}")
    print(f"Output file: {args.output}")
    print(f"Workers: {args.workers}")
    print(f"Retry attempts: {args.retries}")
    
    # Track execution time
    start_time = time.time()
    
    try:
        # Collect the articles
        wiki_df = collect_wikipedia_content(
            count=args.count,
            min_words=args.min_words,
            max_workers=args.workers,
            retry_attempts=args.retries
        )
        
        # Save the dataset
        success, saved_path = save_dataset(wiki_df, filename=args.output)
        
        # Report statistics
        if success:
            total_words = wiki_df['word_count'].sum()
            print(f"\nCollection successful!")
            print(f"Total articles: {len(wiki_df)}")
            print(f"Total words: {total_words:,}")
            print(f"Saved to: {saved_path}")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Saving partial results...")
        if 'wiki_df' in locals() and len(wiki_df) > 0:
            save_dataset(wiki_df, filename=f"partial_{args.output}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Report execution time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nExecution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import wikipediaapi
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia-api", "pandas", "tqdm"])
        print("Required packages installed successfully!")
    
    # Run the main function
    main()