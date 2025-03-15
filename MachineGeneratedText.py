# import numpy as np
# import pandas as pd
# import random
# import time
# import argparse
# import csv
# from typing import List, Dict
# from tqdm import tqdm
# import os
# import re

# # Optional dependencies - will try to use if available
# try:
#     import torch
#     from transformers import pipeline, set_seed
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False

# class WikipediaStyleMarkovGenerator:
#     """Markov chain model specialized for generating Wikipedia-like text."""
    
#     def __init__(self, order=2):
#         self.order = order
#         self.models = {}
#         self.starters = {}
#         self.sentence_enders = {'.', '?', '!'}
#         self.reference_patterns = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]']
        
#     def train_on_wikipedia_corpus(self, corpus_file):
#         """Train on Wikipedia corpus file."""
#         try:
#             with open(corpus_file, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             self.train_on_text(text, "wikipedia")
#             return True
#         except Exception as e:
#             print(f"Error training on Wikipedia corpus: {e}")
#             return False
    
#     def train_on_text(self, text, topic=None):
#         """Train the model on a text."""
#         topic = topic or "wikipedia"
#         if topic not in self.models:
#             self.models[topic] = {}
#             self.starters[topic] = []
        
#         # Split into sentences for better paragraph structure
#         sentences = re.split(r'(?<=[.!?])\s+', text)
#         for sentence in sentences:
#             words = sentence.split()
#             if len(words) <= self.order:
#                 continue
            
#             # Add sentence starters
#             starter = tuple(words[:self.order])
#             self.starters[topic].append(starter)
            
#             # Build model
#             for i in range(len(words) - self.order):
#                 key = tuple(words[i:i+self.order])
#                 if key not in self.models[topic]:
#                     self.models[topic][key] = []
                
#                 if i + self.order < len(words):
#                     self.models[topic][key].append(words[i+self.order])
    
#     def train_on_wikipedia_sample(self):
#         """Train on built-in Wikipedia-style samples."""
#         wikipedia_samples = [
#             """
#             Ancient Rome was a civilization that grew from a city-state founded on the Italian Peninsula circa the 9th century BC to a massive empire straddling the Mediterranean Sea.[1] In its twelve-century existence, Roman civilization shifted from a monarchy to an oligarchic republic to an increasingly autocratic empire. Roman civilization is often grouped into "classical antiquity" with ancient Greece, a civilization that inspired much of the culture of ancient Rome.[2]
            
#             Ancient Roman civilization has contributed to modern language, religion, society, technology, law, politics, government, warfare, art, literature, architecture, and engineering. Rome professionalized and expanded its military and created a system of government called res publica, the inspiration for modern republics[3] such as the United States and France.[4] It achieved impressive technological and architectural feats, such as the construction of an extensive system of aqueducts and roads, as well as large monuments, palaces, and public facilities.
            
#             By the end of the Republic (27 BC), Rome had conquered the lands around the Mediterranean and beyond: its domain extended from the Atlantic to Arabia and from the mouth of the Rhine to North Africa. The Roman Empire emerged under the leadership of Augustus Caesar.[5] Under Trajan, the Empire reached its territorial peak. Republican mores and traditions started to decline during the imperial period, with civil wars becoming a common ritual for a new emperor's rise.[6] States, such as Palmyra, temporarily divided the Empire during the Crisis of the Third Century.[7]
#             """,
#             """
#             Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.[1][2] It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.
            
#             Classical physics, the description of physics that existed before the theory of relativity and quantum mechanics, describes many aspects of nature at an ordinary (macroscopic) scale, while quantum mechanics explains the aspects of nature at small (atomic and subatomic) scales, for which classical mechanics is insufficient.[3] Most theories in classical physics can be derived from quantum mechanics as an approximation valid at large (macroscopic) scale.[4]
            
#             Quantum mechanics differs from classical physics in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to discrete values (quantization), objects have characteristics of both particles and waves (wave-particle duality), and there are limits to how accurately the value of a physical quantity can be predicted prior to its measurement, given a complete set of initial conditions (the uncertainty principle).[5]
#             """,
#             """
#             Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.[3]
            
#             A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[4][5] Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain.[6][7]
            
#             In its application across business problems, machine learning is also referred to as predictive analytics. Machine learning approaches have been applied to large language models, computer vision, speech recognition, email filtering, agriculture, and medicine, where it is too costly to develop algorithms that perform well for all possible inputs.[8]
#             """
#         ]
        
#         for sample in wikipedia_samples:
#             self.train_on_text(sample, "wikipedia")
    
#     def add_wikipedia_formatting(self, text):
#         """Add Wikipedia-style formatting to the text."""
#         # Add section headers
#         sections = ["History", "Overview", "Background", "Development", "Applications", "Criticism", "Legacy", "See also", "References"]
#         formatted_paragraphs = []
        
#         paragraphs = text.split("\n\n")
#         for i, paragraph in enumerate(paragraphs):
#             if i > 0 and random.random() < 0.3:  # 30% chance of adding a section header
#                 section = random.choice(sections)
#                 formatted_paragraphs.append(f"== {section} ==")
            
#             # Add citations
#             sentences = re.split(r'(?<=[.!?])\s+', paragraph)
#             formatted_sentences = []
            
#             for sentence in sentences:
#                 if random.random() < 0.4:  # 40% chance of adding a citation
#                     citation = random.choice(self.reference_patterns)
#                     # Make sure the citation comes after punctuation
#                     if sentence and sentence[-1] in self.sentence_enders:
#                         sentence = sentence[:-1] + citation + sentence[-1]
#                     else:
#                         sentence += citation
#                 formatted_sentences.append(sentence)
            
#             formatted_paragraph = " ".join(formatted_sentences)
#             formatted_paragraphs.append(formatted_paragraph)
        
#         # Add an intro sentence that summarizes the topic
#         if random.random() < 0.8:  # 80% chance
#             first_words = text.split()[:10]
#             topic = " ".join(first_words[:3])
#             intro = f"{topic.capitalize()} is a significant concept that has been studied extensively."
#             formatted_paragraphs.insert(0, intro)
        
#         # Add "References" section at the end
#         if random.random() < 0.7:  # 70% chance
#             formatted_paragraphs.append("== References ==")
#             num_refs = random.randint(3, 10)
#             for i in range(1, num_refs + 1):
#                 author = random.choice(["Smith", "Jones", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson"])
#                 year = random.randint(1980, 2023)
#                 formatted_paragraphs.append(f"[{i}] {author}, {year}. \"Title of the reference\". Journal or Publisher.")
        
#         return "\n\n".join(formatted_paragraphs)
    
#     def generate_wikipedia_style(self, min_words=200, max_words=800):
#         """Generate Wikipedia-style text using the trained model."""
#         topic = "wikipedia"
#         if topic not in self.models or not self.models[topic]:
#             # Fallback to training on sample data
#             self.train_on_wikipedia_sample()
#             if topic not in self.models or not self.models[topic]:
#                 return "No Wikipedia model available."
        
#         if not self.starters[topic]:
#             # No starters found, create random start
#             all_keys = list(self.models[topic].keys())
#             if not all_keys:
#                 return "Model has no data."
#             current = random.choice(all_keys)
#         else:
#             current = random.choice(self.starters[topic])
        
#         words = list(current)
#         target_length = random.randint(min_words, max_words)
        
#         while len(words) < target_length:
#             if current in self.models[topic] and self.models[topic][current]:
#                 next_word = random.choice(self.models[topic][current])
#                 words.append(next_word)
#                 current = tuple(words[-self.order:])
#             else:
#                 # Break in chain, start a new sentence
#                 if self.starters[topic]:
#                     current = random.choice(self.starters[topic])
#                     words.append(".")  # End the previous sentence
#                     words.extend(current)
#                 else:
#                     # No more starters, end generation
#                     break
        
#         raw_text = ' '.join(words)
        
#         # Split into paragraphs for readability
#         paragraph_length = random.randint(3, 7)  # sentences per paragraph
#         sentences = re.split(r'(?<=[.!?])\s+', raw_text)
#         paragraphs = []
        
#         for i in range(0, len(sentences), paragraph_length):
#             paragraph = ' '.join(sentences[i:i+paragraph_length])
#             paragraphs.append(paragraph)
        
#         wiki_text = '\n\n'.join(paragraphs)
        
#         # Add Wikipedia-style formatting
#         wiki_text = self.add_wikipedia_formatting(wiki_text)
        
#         return wiki_text

# def setup_transformers_for_wikipedia():
#     """Setup text generation with HuggingFace Transformers for Wikipedia-style text."""
#     if not TRANSFORMERS_AVAILABLE:
#         return None
    
#     try:
#         # Use a model for text generation
#         generator = pipeline('text-generation', model='distilgpt2')
#         print("Using DistilGPT-2 model for Wikipedia-style text generation")
#         return generator
#     except Exception as e:
#         print(f"Error setting up transformers generator: {e}")
#         return None

# def get_wikipedia_style_prompts(num_prompts: int = 100) -> List[str]:
#     """Generate prompts to guide Wikipedia-style article generation."""
#     encyclopedia_topics = [
#         "History of", "Geography of", "Culture of", "Politics of", "Economy of",
#         "Science in", "Technology in", "Society in", "Art of", "Literature of",
#         "Ancient", "Medieval", "Modern", "Contemporary", "Traditional",
#         "Origins of", "Development of", "Evolution of", "Structure of", "Function of"
#     ]
    
#     subjects = [
#         "Rome", "Greece", "China", "Egypt", "Mesopotamia",
#         "quantum mechanics", "relativity", "thermodynamics", "cell biology", "genetics",
#         "democracy", "socialism", "capitalism", "feudalism", "monarchy",
#         "Renaissance", "Industrial Revolution", "World War II", "Cold War", "French Revolution",
#         "impressionism", "cubism", "romanticism", "modernism", "baroque",
#         "artificial intelligence", "internet", "blockchain", "computing", "robotics"
#     ]
    
#     prompts = []
#     for _ in range(num_prompts):
#         topic = random.choice(encyclopedia_topics)
#         subject = random.choice(subjects)
#         prompt = f"{topic} {subject}"
#         prompts.append(prompt)
    
#     return prompts

# def generate_wikipedia_style_text(
#     generator,
#     prompts: List[str],
#     count: int = 1000,
#     min_words: int = 200,
#     max_words: int = 800,
#     use_transformers: bool = True
# ) -> List[Dict]:
#     """Generate Wikipedia-style machine-written text."""
#     generated_articles = []
    
#     # Always prepare the Markov generator as fallback
#     markov_generator = WikipediaStyleMarkovGenerator(order=2)
#     markov_generator.train_on_wikipedia_sample()
    
#     # Setup progress tracking
#     pbar = tqdm(total=count, desc="Generating Wikipedia-style articles")
    
#     # Repeat prompts if necessary to reach desired count
#     expanded_prompts = []
#     while len(expanded_prompts) < count:
#         expanded_prompts.extend(prompts)
#     expanded_prompts = expanded_prompts[:count]
    
#     # Generate the articles
#     for i, prompt in enumerate(expanded_prompts[:count]):
#         try:
#             if use_transformers and generator is not None:
#                 # Use transformers pipeline with Wikipedia-specific prompting
#                 prompt_text = f"Write a Wikipedia article about {prompt}:"
#                 result = generator(
#                     prompt_text,
#                     max_length=min(100, min_words // 4),
#                     do_sample=True,
#                     temperature=0.9
#                 )
#                 text = result[0]['generated_text']
                
#                 # Remove prompt from generated text
#                 if prompt_text in text:
#                     text = text[len(prompt_text):].strip()
                
#                 # If text is too short, extend it
#                 words = text.split()
#                 if len(words) < min_words:
#                     additional_iterations = (min_words - len(words)) // 50 + 1
#                     for _ in range(additional_iterations):
#                         continuation = generator(
#                             text[-100:], 
#                             max_length=50,
#                             do_sample=True,
#                             temperature=0.9
#                         )
#                         text += " " + continuation[0]['generated_text']
#                         words = text.split()
#                         if len(words) >= min_words:
#                             break
                
#                 # Truncate if too long
#                 if len(words) > max_words:
#                     text = " ".join(words[:max_words])
                
#                 # Apply Wikipedia formatting to the transformers output
#                 text = markov_generator.add_wikipedia_formatting(text)
#             else:
#                 # Use Markov chain for Wikipedia-style text
#                 text = markov_generator.generate_wikipedia_style(
#                     min_words=min_words,
#                     max_words=max_words
#                 )
            
#             # Add to collection
#             generated_articles.append({
#                 "text": text,
#                 "prompt": prompt,
#                 "label": 1  # Label 1 for machine-generated text
#             })
            
#         except Exception as e:
#             print(f"Error generating Wikipedia article {i}: {e}")
#             # Add placeholder with Wikipedia style
#             generated_articles.append({
#                 "text": f"{prompt} is a subject of significant interest in various fields of study.[1] The development of {prompt} has been documented extensively in academic literature.[2] Various theories exist regarding the origins and implications of {prompt}.[3] Further research is needed to fully understand its impact.",
#                 "prompt": prompt,
#                 "label": 1
#             })
        
#         pbar.update(1)
        
#         # Occasionally clear CUDA cache if using GPU
#         if use_transformers and i % 800 == 0 and 'torch' in globals():
#             try:
#                 torch.cuda.empty_cache()
#             except:
#                 pass
    
#     pbar.close()
#     return generated_articles

# def main():
#     """Main function to run the Wikipedia-style text generator."""
#     parser = argparse.ArgumentParser(description="Generate Wikipedia-style machine text dataset")
#     parser.add_argument("--count", type=int, default=1000, help="Number of Wikipedia-style articles to generate")
#     parser.add_argument("--output", type=str, default="wikipedia_machine_text.csv", help="Output CSV file path")
#     parser.add_argument("--min_words", type=int, default=200, help="Minimum words per text")
#     parser.add_argument("--max_words", type=int, default=800, help="Maximum words per text")
#     parser.add_argument("--no_transformers", action="store_true", help="Don't use transformers even if available")
#     parser.add_argument("--wikipedia_corpus", type=str, default=None, help="Path to Wikipedia corpus file (optional)")
    
#     args = parser.parse_args()
    
#     try:
#         print("Setting up Wikipedia-style text generator...")
        
#         # Prepare for generation
#         use_transformers = TRANSFORMERS_AVAILABLE and not args.no_transformers
        
#         # Setup generator
#         generator = None
#         if use_transformers:
#             generator = setup_transformers_for_wikipedia()
#             use_transformers = generator is not None
        
#         # Generate prompts
#         print("Generating Wikipedia-style prompts...")
#         prompts = get_wikipedia_style_prompts(100)
        
#         # Setup Markov generator with Wikipedia corpus if provided
#         markov_generator = WikipediaStyleMarkovGenerator(order=2)
#         if args.wikipedia_corpus:
#             print(f"Training on Wikipedia corpus: {args.wikipedia_corpus}")
#             success = markov_generator.train_on_wikipedia_corpus(args.wikipedia_corpus)
#             if not success:
#                 print("Falling back to sample Wikipedia data...")
#                 markov_generator.train_on_wikipedia_sample()
#         else:
#             print("Using built-in Wikipedia samples for Markov model...")
#             markov_generator.train_on_wikipedia_sample()
        
#         # Generate Wikipedia-style machine text
#         print(f"Generating {args.count} Wikipedia-style machine-written texts...")
#         wiki_machine_texts = generate_wikipedia_style_text(
#             generator,
#             prompts=prompts,
#             count=args.count,
#             min_words=args.min_words,
#             max_words=args.max_words,
#             use_transformers=use_transformers
#         )
        
#         # Export to CSV
#         export_to_csv(wiki_machine_texts, args.output)
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# def export_to_csv(articles: List[Dict], output_file: str = "wikipedia_machine_text_with_1000_As_Human.csv"):
#     """Export articles to a CSV file."""
#     try:
#         # Write to CSV
#         with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#             fieldnames = ['text', 'label']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
#             writer.writeheader()
#             for article in articles:
#                 writer.writerow({
#                     'text': article['text'],
#                     'label': article['label']
#                 })
        
#         print(f"Successfully exported {len(articles)} articles to {output_file}")
        
#     except Exception as e:
#         print(f"Error exporting to CSV: {str(e)}")
        
#         # Fallback: Try to save as a JSON file
#         try:
#             import json
#             json_path = output_file.replace('.csv', '.json')
#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(articles, f, ensure_ascii=False, indent=2)
#             print(f"Exported as JSON instead at: {json_path}")
#         except Exception as json_err:
#             print(f"Failed to export as JSON as well: {str(json_err)}")

# if __name__ == "__main__":
#     main()



############################################################################




import numpy as np
import pandas as pd
import random
import time
import argparse
import csv
from typing import List, Dict
import os
import re
import json
import requests
from urllib.parse import quote

class WikipediaStyleGenerator:
    """Generator for Wikipedia-style text that resembles human writing."""
    
    def __init__(self, order=2):
        self.order = order
        self.models = {}
        self.starters = {}
        self.sentence_enders = {'.', '?', '!'}
        self.reference_patterns = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]']
        self.train_on_wikipedia_sample()
        
    def train_on_text(self, text, topic=None):
        """Train the model on a text."""
        topic = topic or "wikipedia"
        if topic not in self.models:
            self.models[topic] = {}
            self.starters[topic] = []
        
        # Split into sentences for better paragraph structure
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            words = sentence.split()
            if len(words) <= self.order:
                continue
            
            # Add sentence starters
            starter = tuple(words[:self.order])
            self.starters[topic].append(starter)
            
            # Build model
            for i in range(len(words) - self.order):
                key = tuple(words[i:i+self.order])
                if key not in self.models[topic]:
                    self.models[topic][key] = []
                
                if i + self.order < len(words):
                    self.models[topic][key].append(words[i+self.order])
    
    def train_on_wikipedia_sample(self):
        """Train on built-in Wikipedia-style samples."""
        wikipedia_samples = [
            """
            Ancient Rome was a civilization that grew from a city-state founded on the Italian Peninsula circa the 9th century BC to a massive empire straddling the Mediterranean Sea.[1] In its twelve-century existence, Roman civilization shifted from a monarchy to an oligarchic republic to an increasingly autocratic empire. Roman civilization is often grouped into "classical antiquity" with ancient Greece, a civilization that inspired much of the culture of ancient Rome.[2]
            
            Ancient Roman civilization has contributed to modern language, religion, society, technology, law, politics, government, warfare, art, literature, architecture, and engineering. Rome professionalized and expanded its military and created a system of government called res publica, the inspiration for modern republics[3] such as the United States and France.[4] It achieved impressive technological and architectural feats, such as the construction of an extensive system of aqueducts and roads, as well as large monuments, palaces, and public facilities.
            
            By the end of the Republic (27 BC), Rome had conquered the lands around the Mediterranean and beyond: its domain extended from the Atlantic to Arabia and from the mouth of the Rhine to North Africa. The Roman Empire emerged under the leadership of Augustus Caesar.[5] Under Trajan, the Empire reached its territorial peak. Republican mores and traditions started to decline during the imperial period, with civil wars becoming a common ritual for a new emperor's rise.[6] States, such as Palmyra, temporarily divided the Empire during the Crisis of the Third Century.[7]
            """,
            """
            Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.[1][2] It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.
            
            Classical physics, the description of physics that existed before the theory of relativity and quantum mechanics, describes many aspects of nature at an ordinary (macroscopic) scale, while quantum mechanics explains the aspects of nature at small (atomic and subatomic) scales, for which classical mechanics is insufficient.[3] Most theories in classical physics can be derived from quantum mechanics as an approximation valid at large (macroscopic) scale.[4]
            
            Quantum mechanics differs from classical physics in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to discrete values (quantization), objects have characteristics of both particles and waves (wave-particle duality), and there are limits to how accurately the value of a physical quantity can be predicted prior to its measurement, given a complete set of initial conditions (the uncertainty principle).[5]
            """,
            """
            Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.[3]
            
            A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[4][5] Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain.[6][7]
            
            In its application across business problems, machine learning is also referred to as predictive analytics. Machine learning approaches have been applied to large language models, computer vision, speech recognition, email filtering, agriculture, and medicine, where it is too costly to develop algorithms that perform well for all possible inputs.[8]
            """
        ]
        
        # Additional samples for more variety
        additional_samples = [
            """
            The Solar System is the gravitationally bound system of the Sun and the objects that orbit it, either directly or indirectly.[1] Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, the dwarf planets and small Solar System bodies. Of the objects that orbit the Sun indirectly—the natural satellites—two are larger than the smallest planet, Mercury.[2]
            
            The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in Jupiter. The four smaller inner planets, Mercury, Venus, Earth and Mars, are terrestrial planets, being primarily composed of rock and metal.[3] The four outer planets are giant planets, being substantially more massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; the two outermost planets, Uranus and Neptune, are ice giants, being composed mostly of substances with relatively high melting points compared with hydrogen and helium, called volatiles, such as water, ammonia and methane.[4] All eight planets have almost circular orbits that lie within a nearly flat disc called the ecliptic.
            """,
            """
            The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799.[1] Many of its ideas are considered fundamental principles of liberal democracy, while phrases like "Liberty, Equality, Fraternity" reappeared in other revolts, such as the 1917 Russian Revolution, and inspired campaigns for the abolition of slavery and universal suffrage.[2] The values and institutions it created dominate French politics to this day.

            Its causes are generally agreed to be a combination of social, political and economic factors, which the existing regime proved unable to manage.[3] In May 1789, widespread social distress led to the convocation of the Estates General, which was converted into a National Assembly in June.[4] Continuing unrest culminated in the Storming of the Bastille on 14 July, which led to a series of radical measures by the Assembly, including the abolition of feudalism, the imposition of state control over the Catholic Church in France, and extension of the right to vote.[5]
            """,
            """
            Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.[1] AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.[2]

            The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".[3] This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.[4]

            AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative AI, and competing at the highest level in strategic game systems.[5] As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.[6] For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.[7]
            """
        ]
        
        wikipedia_samples.extend(additional_samples)
        
        for sample in wikipedia_samples:
            self.train_on_text(sample, "wikipedia")
    
    def add_wikipedia_formatting(self, text, topic):
        """Add Wikipedia-style formatting to the text."""
        # Add section headers
        sections = [
            "History", "Overview", "Background", "Development", "Applications", 
            "Criticism", "Legacy", "Theory", "Practice", "Modern interpretations",
            "Scientific perspective", "Cultural impact", "Economic significance",
            "Environmental considerations", "Political consequences", "Ethical implications",
            "Technological advancements", "Global significance", "Regional variations",
            "Academic research", "Popular understanding", "Future directions"
        ]
        
        # Shuffle sections to ensure we don't always get the same order
        random.shuffle(sections)
        formatted_paragraphs = []
        
        # Add title
        title = f"# {topic.title()}"
        formatted_paragraphs.append(title)
        
        # Add introduction paragraph
        intro = f"{topic.title()} is a significant concept that has been studied extensively across multiple disciplines.[1] The development of research in this area has led to numerous breakthroughs in understanding its fundamental principles and applications.[2]"
        formatted_paragraphs.append(intro)
        
        # Process paragraphs
        paragraphs = text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            if i > 0 and random.random() < 0.4:  # 40% chance of adding a section header
                section = sections.pop(0) if sections else "Additional information"
                formatted_paragraphs.append(f"## {section}")
            
            # Add citations
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            formatted_sentences = []
            
            for sentence in sentences:
                if not sentence:
                    continue
                    
                if random.random() < 0.4:  # 40% chance of adding a citation
                    citation = random.choice(self.reference_patterns)
                    # Make sure the citation comes after punctuation
                    if sentence and sentence[-1] in self.sentence_enders:
                        sentence = sentence[:-1] + citation + sentence[-1]
                    else:
                        sentence += citation
                formatted_sentences.append(sentence)
            
            formatted_paragraph = " ".join(formatted_sentences)
            formatted_paragraphs.append(formatted_paragraph)
        
        # Add "References" section at the end
        if random.random() < 0.9:  # 90% chance
            formatted_paragraphs.append("## References")
            num_refs = random.randint(3, 10)
            for i in range(1, num_refs + 1):
                author = random.choice([
                    "Smith", "Jones", "Johnson", "Williams", "Brown", "Davis", "Miller", 
                    "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", 
                    "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"
                ])
                year = random.randint(1980, 2023)
                titles = [
                    f"Understanding {topic}",
                    f"A Comprehensive Analysis of {topic}",
                    f"The Theory and Practice of {topic}",
                    f"{topic}: A New Perspective",
                    f"Advancements in {topic} Research",
                    f"The Evolution of {topic} Studies",
                    f"{topic} in Contemporary Context",
                    f"Exploring the Fundamentals of {topic}",
                    f"Critical Approaches to {topic}",
                    f"Rethinking {topic}"
                ]
                journals = [
                    "Journal of Advanced Studies", 
                    "International Review", 
                    "Academic Quarterly",
                    "Science Today",
                    "Modern Research",
                    "Theoretical Studies Journal",
                    "Applied Research Review",
                    "Contemporary Analysis",
                    "Global Research Perspectives",
                    "Interdisciplinary Studies"
                ]
                formatted_paragraphs.append(f"[{i}] {author}, {year}. \"{random.choice(titles)}\". {random.choice(journals)}, vol. {random.randint(1, 50)}, pp. {random.randint(1, 300)}-{random.randint(301, 500)}.")
        
        return "\n\n".join(formatted_paragraphs)
    
    def generate_unique_topics(self, count=1000):
        """Generate unique topics for Wikipedia articles."""
        prefixes = [
            "History of", "Theory of", "Introduction to", "Principles of", 
            "Development of", "Origins of", "Applications of", "Studies in", 
            "Concepts in", "Perspectives on", "Advances in", "Fundamentals of", 
            "Overview of", "Analysis of", "Exploration of", "Understanding", 
            "Research on", "Approaches to", "Examination of", "Critical view of"
        ]
        
        domains = [
            "physics", "chemistry", "biology", "mathematics", "computer science",
            "philosophy", "psychology", "sociology", "anthropology", "economics",
            "political science", "history", "literature", "linguistics", "arts",
            "medicine", "astronomy", "geology", "environmental science", "neuroscience",
            "archaeology", "geography", "education", "engineering", "architecture",
            "music", "film studies", "media studies", "communication", "law",
            "ethics", "religion", "cultural studies", "gender studies", "urban planning",
            "public health", "information science", "cognitive science", "game theory",
            "cybernetics", "systems theory", "quantum theory", "relativity", "thermodynamics",
            "evolutionary biology", "genetics", "ecology", "marine biology", "botany",
            "artificial intelligence", "machine learning", "data science", "robotics", "cryptography"
        ]
        
        specific_subjects = [
            "renaissance", "enlightenment", "industrial revolution", "modernism",
            "romanticism", "classical period", "baroque era", "ancient civilizations",
            "medieval period", "cold war", "globalization", "digital age", "space exploration",
            "climate change", "sustainable development", "technological innovation",
            "democratic institutions", "market economies", "social movements", "human rights",
            "artistic expression", "cultural identity", "psychological development",
            "cognitive processes", "economic growth", "scientific method", "theoretical frameworks",
            "empirical research", "qualitative analysis", "quantitative methods",
            "urban development", "rural communities", "indigenous knowledge", "western philosophy",
            "eastern thought", "comparative studies", "interdisciplinary approaches",
            "practical applications", "theoretical implications", "historical context",
            "contemporary issues", "future trends", "emerging technologies", "social impact",
            "ethical considerations", "legal frameworks", "policy development",
            "institutional structures", "individual behavior", "collective action",
            "creative expression", "critical thinking", "problem solving", "decision making"
        ]
        
        # Create combinations
        topics = []
        used_topics = set()
        
        # Generate combined topics first
        for _ in range(count // 2):
            prefix = random.choice(prefixes)
            domain = random.choice(domains)
            topic = f"{prefix} {domain}"
            
            if topic not in used_topics:
                topics.append(topic)
                used_topics.add(topic)
        
        # Generate domain-specific topics
        for _ in range(count // 4):
            domain = random.choice(domains)
            subject = random.choice(specific_subjects)
            topic = f"{domain} in {subject}"
            
            if topic not in used_topics:
                topics.append(topic)
                used_topics.add(topic)
                
        # Generate direct domain topics
        remaining = count - len(topics)
        for _ in range(remaining):
            if random.random() < 0.5:
                topic = random.choice(domains)
            else:
                topic = random.choice(specific_subjects)
                
            if topic not in used_topics:
                topics.append(topic)
                used_topics.add(topic)
        
        # If we still don't have enough, create more combinations
        while len(topics) < count:
            prefix = random.choice(prefixes)
            domain = random.choice(domains) if random.random() < 0.5 else random.choice(specific_subjects)
            topic = f"{prefix} {domain}"
            
            if topic not in used_topics:
                topics.append(topic)
                used_topics.add(topic)
        
        # Shuffle and return only the number requested
        random.shuffle(topics)
        return topics[:count]
    
    def generate_wikipedia_style(self, topic, min_words=250, max_words=800):
        """Generate Wikipedia-style text that is actually about the given topic."""
        # Extract the core subject from the topic
        topic_parts = topic.split()
        core_subject = topic_parts[-1] if topic_parts else topic
        
        # Get dynamic templates from external APIs
        dynamic_templates = self.fetch_dynamic_templates(topic, core_subject)
        
        # Generate multiple paragraphs about the topic
        paragraphs = []
        
        # Start with a dynamic template as the first paragraph
        paragraphs.append(random.choice(dynamic_templates))
        
        # Generate topic-related terms
        related_terms = self.generate_related_terms(topic)
        
        # Create domain-specific paragraph templates
        if "physics" in topic.lower() or "quantum" in topic.lower() or "relativity" in topic.lower():
            paragraphs.append(f"In the domain of physics, {topic} involves the study of fundamental properties and behaviors of matter and energy. Experimental evidence has consistently supported theoretical predictions, although several areas remain under active investigation. Recent advancements in measurement techniques have allowed for more precise observations of phenomena related to {related_terms[0]} and {related_terms[1]}.")
        
        elif "biology" in topic.lower() or "genetics" in topic.lower() or "evolution" in topic.lower():
            paragraphs.append(f"Biological systems demonstrate remarkable complexity in the context of {topic}. The interactions between {related_terms[0]} and {related_terms[1]} create emergent properties that cannot be predicted from individual components alone. Evolutionary perspectives have provided valuable insights into how these systems developed over time and continue to adapt to changing environments.")
        
        elif "computer" in topic.lower() or "artificial intelligence" in topic.lower() or "machine learning" in topic.lower():
            paragraphs.append(f"Computational approaches to {topic} have revolutionized how researchers analyze and model complex systems. Algorithms based on {related_terms[0]} principles can efficiently process large datasets and identify patterns that would be difficult to detect through traditional methods. The integration of {related_terms[1]} with existing frameworks has opened new avenues for both theoretical advancement and practical applications.")
        
        elif "psychology" in topic.lower() or "cognitive" in topic.lower():
            paragraphs.append(f"Psychological research on {topic} examines both cognitive and affective dimensions of human experience. Studies have demonstrated significant individual differences in how people process information related to {related_terms[0]}. Cultural factors also play an important role in shaping perceptions and behaviors associated with {related_terms[1]}.")
        
        elif "history" in topic.lower() or "ancient" in topic.lower() or "medieval" in topic.lower():
            paragraphs.append(f"Historical analyses of {topic} reveal complex patterns of continuity and change across different time periods and geographical regions. Primary sources provide valuable evidence about how {related_terms[0]} was understood and practiced in various cultural contexts. Scholarly interpretations have evolved as new methodological approaches and theoretical frameworks have been applied to the study of {related_terms[1]}.")
        
        elif "economics" in topic.lower() or "market" in topic.lower():
            paragraphs.append(f"Economic dimensions of {topic} highlight the interplay between individual decision-making and broader structural factors. Market mechanisms influence how resources related to {related_terms[0]} are allocated and utilized. Policy interventions designed to address inefficiencies or inequities must account for complex feedback loops and potential unintended consequences related to {related_terms[1]}.")
        
        elif "philosophy" in topic.lower() or "ethics" in topic.lower():
            paragraphs.append(f"Philosophical inquiry into {topic} raises fundamental questions about knowledge, reality, and value. Epistemological concerns relate to how we can establish reliable knowledge about {related_terms[0]}. Ethical considerations focus on normative dimensions of {related_terms[1]}, including questions of rights, responsibilities, and the common good.")
        
        else:
            # Generic academic paragraph for other topics
            paragraphs.append(f"Academic discourse surrounding {topic} encompasses diverse theoretical perspectives and methodological approaches. Quantitative studies have documented statistical relationships between {related_terms[0]} and various outcome measures. Qualitative research provides rich contextual understanding of how {related_terms[1]} is experienced and interpreted by different stakeholders.")
        
        # Add methodology paragraph
        methodology_templates = [
            f"Research methodologies in {topic} have evolved to address the complex nature of the subject matter. Early studies relied primarily on observational techniques, while contemporary approaches incorporate advanced statistical methods and computational modeling. Methodological triangulation—using multiple techniques to study the same phenomenon—has become increasingly common in investigations of {related_terms[2]}.",
            
            f"The empirical study of {topic} presents unique methodological challenges. Researchers have developed specialized instruments to measure key variables related to {related_terms[0]}. Validity and reliability concerns are addressed through rigorous testing and refinement of these measurement tools. Longitudinal designs are particularly valuable for understanding developmental trajectories and causal relationships in {related_terms[2]}.",
            
            f"Data collection in {topic} research spans multiple levels of analysis, from micro-level processes to macro-level patterns. Interdisciplinary collaborations have facilitated the integration of methods from diverse fields such as {related_terms[1]} and {related_terms[2]}. Ethical considerations guide research design and implementation, particularly when studying sensitive aspects of human experience or vulnerable populations."
        ]
        
        paragraphs.append(random.choice(methodology_templates))
        
        # Add findings/implications paragraph
        findings_templates = [
            f"Findings from research on {topic} have significant implications for both theory and practice. Evidence consistently indicates that {related_terms[0]} influences outcomes through multiple pathways. However, contextual factors moderate these relationships, suggesting the importance of tailored approaches rather than one-size-fits-all solutions. Future directions include more nuanced examination of how {related_terms[1]} and {related_terms[2]} interact across different settings and populations.",
            
            f"The body of knowledge regarding {topic} points to several robust conclusions. First, {related_terms[0]} demonstrates considerable variability across different contexts and populations. Second, interventions targeting {related_terms[1]} have shown promise but require careful implementation and evaluation. Third, theoretical models must account for dynamic interactions between individual and environmental factors related to {related_terms[2]}.",
            
            f"Empirical evidence regarding {topic} has accumulated rapidly in recent years. Meta-analyses indicate moderate to strong effects of {related_terms[0]} on key outcome measures. However, publication bias and methodological limitations necessitate cautious interpretation of these findings. Replication efforts and open science practices are increasingly prioritized to strengthen the evidentiary foundation of research on {related_terms[1]} and {related_terms[2]}."
        ]
        
        paragraphs.append(random.choice(findings_templates))
        
        # Add future directions
        future_templates = [
            f"Future directions in {topic} research include several promising avenues. Technological innovations will enable more precise measurement and analysis of phenomena related to {related_terms[0]}. Interdisciplinary collaborations will continue to enrich theoretical frameworks by incorporating insights from adjacent fields studying {related_terms[1]}. Applied research will focus on translating basic science findings into practical interventions and policies that address real-world challenges related to {related_terms[2]}.",
            
            f"The field of {topic} continues to evolve, with several emerging trends shaping its trajectory. Increased emphasis on ecological validity is improving the relevance of findings to real-world contexts involving {related_terms[0]}. Methodological pluralism acknowledges the value of diverse approaches to understanding complex phenomena related to {related_terms[1]}. Participatory research paradigms are expanding opportunities for collaboration between researchers and communities affected by issues related to {related_terms[2]}.",
            
            f"Looking ahead, several challenges and opportunities will influence the development of {topic} as a field of inquiry. Balancing specialization with integration remains an ongoing tension in research on {related_terms[0]}. Addressing disparities in access and outcomes related to {related_terms[1]} will require both rigorous research and thoughtful implementation. Technological advancements will create new possibilities for understanding and addressing complex problems in the domain of {related_terms[2]}."
        ]
        
        paragraphs.append(random.choice(future_templates))
        
        # Combine paragraphs with formatting
        wiki_text = "\n\n".join(paragraphs)
        
        # Ensure minimum word count
        current_word_count = len(wiki_text.split())
        if current_word_count < min_words:
            # Add more topic-specific content
            additional_templates = [
                f"Critical debates within the field of {topic} center on several key issues. Some scholars emphasize the importance of {related_terms[0]} as a fundamental concept, while others focus more on practical applications related to {related_terms[1]}. These discussions reflect broader tensions between theoretical and applied approaches to knowledge production and utilization.",
                
                f"The historical development of {topic} reflects changing intellectual and social contexts. Early work focused primarily on establishing basic principles and documenting core phenomena related to {related_terms[0]}. As the field matured, attention shifted to more complex questions about mechanisms and processes underlying {related_terms[1]}. Contemporary approaches increasingly incorporate perspectives from diverse cultural and disciplinary traditions.",
                
                f"Pedagogical approaches to {topic} have evolved alongside research developments. Educational programs now emphasize both theoretical understanding and practical skills related to {related_terms[0]}. Experiential learning opportunities allow students to engage directly with challenges and opportunities in the domain of {related_terms[1]}. Professional development continues throughout careers as new findings and methodologies emerge."
            ]
            
            # Add paragraphs until we reach minimum word count
            while current_word_count < min_words:
                new_paragraph = random.choice(additional_templates)
                wiki_text += "\n\n" + new_paragraph
                current_word_count = len(wiki_text.split())
        
        # Limit to maximum word count if needed
        if current_word_count > max_words:
            words = wiki_text.split()
            wiki_text = " ".join(words[:max_words])
        
        # Add Wikipedia-style formatting
        wiki_text = self.add_wikipedia_formatting(wiki_text, topic)
        
        return wiki_text
        
    def fetch_dynamic_templates(self, topic, core_subject):
        """Fetch dynamic templates from free APIs for the given topic."""
        # Initialize with some fallback templates in case API calls fail
        fallback_templates = [
            f"{topic} is a field of study that examines the relationship between {core_subject} and broader societal contexts. Researchers in this area have developed numerous methodologies to understand the underlying principles and practical applications.",
            f"The development of {topic} can be traced back to early investigations in the mid-20th century. Pioneering work by several researchers established the foundation for what would eventually become a distinct discipline.",
            f"Studies in {topic} have evolved significantly over the past decades, with advances in both theoretical understanding and empirical methods."
        ]
        
        try:
            # Try to get content from Wikipedia API
            wiki_templates = self.fetch_from_wikipedia_api(topic)
            if wiki_templates:
                return wiki_templates
                
            # Try to get content from Datamuse API for related terms
            datamuse_templates = self.generate_templates_from_datamuse(topic, core_subject)
            if datamuse_templates:
                return datamuse_templates
                
            # Try to get content from Conceptnet API
            conceptnet_templates = self.fetch_from_conceptnet(topic, core_subject)
            if conceptnet_templates:
                return conceptnet_templates
                
        except Exception as e:
            print(f"Error fetching dynamic templates: {e}")
            
        # Return fallback templates if all API calls fail
        return fallback_templates
    
    def fetch_from_wikipedia_api(self, topic):
        """Fetch content from Wikipedia API and convert to templates."""
        topic_encoded = quote(topic)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic_encoded}"
        
        templates = []
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Extract useful text from the summary
                if 'extract' in data:
                    extract = data['extract']
                    # Split into sentences and create templates
                    sentences = re.split(r'(?<=[.!?])\s+', extract)
                    
                    # Create templates from the first few sentences
                    if len(sentences) >= 3:
                        templates.append(sentences[0])
                        
                        # Create a second template combining other sentences
                        if len(sentences) >= 5:
                            second_template = " ".join(sentences[1:3])
                            templates.append(second_template)
                            
                            third_template = " ".join(sentences[3:5])
                            templates.append(third_template)
                        
                # If we have a description, use it as another template
                if 'description' in data:
                    desc_template = f"{topic} is {data['description']}. This field has seen significant development in recent years with advances in both theory and practice."
                    templates.append(desc_template)
        except Exception as e:
            print(f"Wikipedia API error: {e}")
            
        return templates
    
    def generate_templates_from_datamuse(self, topic, core_subject):
        """Generate templates using related words from Datamuse API."""
        templates = []
        
        try:
            # Get related words using Datamuse API
            url = f"https://api.datamuse.com/words?ml={quote(topic)}&max=10"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                related_words = response.json()
                
                if related_words:
                    # Extract word strings
                    words = [item['word'] for item in related_words if 'word' in item]
                    
                    if len(words) >= 3:
                        # Create templates using related words
                        templates.append(f"{topic} is closely related to concepts such as {', '.join(words[:3])}. Research in this area has explored these connections to develop a more comprehensive understanding of {core_subject}.")
                        
                    if len(words) >= 6:
                        templates.append(f"The field of {topic} encompasses various aspects including {words[3]}, {words[4]}, and {words[5]}. These components interact in complex ways that shape both theoretical frameworks and practical applications.")
                        
                    # Get synonyms for additional context
                    syn_url = f"https://api.datamuse.com/words?rel_syn={quote(core_subject)}&max=5"
                    syn_response = requests.get(syn_url, timeout=5)
                    
                    if syn_response.status_code == 200:
                        synonyms = syn_response.json()
                        syn_words = [item['word'] for item in synonyms if 'word' in item]
                        
                        if syn_words:
                            templates.append(f"In academic discourse, {topic} is sometimes referred to in terms of {', '.join(syn_words[:3])}. These alternative framings provide different perspectives on the core phenomena under investigation.")
        except Exception as e:
            print(f"Datamuse API error: {e}")
            
        return templates
    
    def fetch_from_conceptnet(self, topic, core_subject):
        """Fetch related concepts from ConceptNet API and create templates."""
        templates = []
        
        try:
            # Clean up topic for API query
            query_term = topic.lower().replace(" ", "_")
            url = f"https://api.conceptnet.io/c/en/{query_term}"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'edges' in data:
                    # Extract relevant relationships
                    relations = []
                    for edge in data['edges'][:15]:  # Limit to first 15 edges
                        if 'rel' in edge and 'label' in edge['rel']:
                            rel_type = edge['rel']['label']
                            
                            target = None
                            if 'end' in edge and 'label' in edge['end']:
                                target = edge['end']['label']
                            
                            if target and rel_type:
                                relations.append((rel_type, target))
                    
                    # Create templates from relationships
                    if relations:
                        rel_template = f"{topic} has several important connections in its conceptual framework. "
                        
                        # Add specific relationships
                        related_concepts = []
                        for rel_type, target in relations[:3]:
                            if rel_type == "IsA":
                                related_concepts.append(f"it is a type of {target}")
                            elif rel_type == "HasA":
                                related_concepts.append(f"it possesses {target}")
                            elif rel_type == "UsedFor":
                                related_concepts.append(f"it is used for {target}")
                            elif rel_type == "CapableOf":
                                related_concepts.append(f"it is capable of {target}")
                            elif rel_type == "RelatedTo":
                                related_concepts.append(f"it is related to {target}")
                                
                        if related_concepts:
                            rel_template += f"Specifically, {'; '.join(related_concepts)}."
                            templates.append(rel_template)
                            
                        # Create a second template with additional concepts
                        if len(relations) > 3:
                            property_relations = [target for rel_type, target in relations[3:6] 
                                               if rel_type in ["HasProperty", "HasA", "PartOf", "MadeOf"]]
                            
                            if property_relations:
                                prop_template = f"The core characteristics of {topic} include {', '.join(property_relations)}. These properties define how {topic} is understood and applied in various contexts."
                                templates.append(prop_template)
        except Exception as e:
            print(f"ConceptNet API error: {e}")
            
        return templates
            
    def generate_related_terms(self, topic):
        """Generate terms related to the given topic for more realistic content."""
        topic_lower = topic.lower()
        
        # Default generic terms
        generic_terms = [
            "theoretical frameworks", "empirical evidence", "practical applications",
            "methodological approaches", "critical perspectives", "historical development",
            "foundational principles", "key concepts", "analytical techniques",
            "interdisciplinary connections", "contemporary challenges", "future directions"
        ]
        
        # Domain-specific related terms
        domain_terms = {
            "physics": ["quantum phenomena", "relativistic effects", "field theories", 
                       "particle interactions", "wave functions", "symmetry principles",
                       "conservation laws", "experimental methods", "theoretical models",
                       "measurement precision"],
                       
            "biology": ["cellular mechanisms", "evolutionary processes", "genetic factors",
                       "ecological interactions", "physiological systems", "developmental pathways",
                       "molecular structures", "taxonomic classifications", "biodiversity patterns",
                       "adaptive responses"],
                       
            "computer": ["algorithmic efficiency", "computational complexity", "data structures",
                        "software architecture", "hardware systems", "network protocols",
                        "information security", "user interfaces", "artificial intelligence",
                        "machine learning models"],
                        
            "psychology": ["cognitive processes", "behavioral patterns", "emotional responses",
                          "developmental stages", "personality factors", "social influences",
                          "clinical interventions", "assessment methods", "neurological correlates",
                          "cultural variations"],
                          
            "history": ["primary sources", "archaeological evidence", "chronological developments",
                       "cultural contexts", "political structures", "economic systems",
                       "social hierarchies", "intellectual movements", "geographical factors",
                       "comparative perspectives"],
                       
            "economics": ["market dynamics", "fiscal policies", "monetary systems",
                         "resource allocation", "production factors", "consumption patterns",
                         "distribution mechanisms", "growth models", "equilibrium states",
                         "behavioral incentives"],
                         
            "philosophy": ["ontological questions", "epistemological frameworks", "ethical principles",
                          "logical structures", "metaphysical concepts", "phenomenological analyses",
                          "existential concerns", "analytical methods", "conceptual distinctions",
                          "normative claims"]
        }
        
        # Find matching domain
        selected_terms = generic_terms.copy()
        for domain, terms in domain_terms.items():
            if domain in topic_lower:
                selected_terms.extend(terms)
                break
                
        # Get 3 unique random terms
        if len(selected_terms) > 3:
            return random.sample(selected_terms, 3)
        else:
            # Pad with generic terms if needed
            result = selected_terms.copy()
            additional_needed = 3 - len(result)
            if additional_needed > 0:
                additional_terms = random.sample(generic_terms, additional_needed)
                result.extend(additional_terms)
            return result

def generate_and_export_articles(count=1000, min_words=250, max_words=800, output_file="wikipedia_style_text.csv"):
    """Generate and export Wikipedia-style articles."""
    generator = WikipediaStyleGenerator(order=2)
    
    # Generate unique topics
    topics = generator.generate_unique_topics(count)
    
    # Generate articles
    articles = []
    print(f"Generating {count} Wikipedia-style articles...")
    
    for i, topic in enumerate(topics):
        if i % 100 == 0 and i > 0:
            print(f"Generated {i} articles...")
        
        article = generator.generate_wikipedia_style(
            topic=topic,
            min_words=min_words,
            max_words=max_words
        )
        
        articles.append({
            "text": article,
            "topic": topic,
            "label": 1  # Label 1 for machine-generated text (similar to human)
        })
    
    # Export to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'topic', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for article in articles:
            writer.writerow(article)
    
    print(f"Successfully exported {len(articles)} articles to {output_file}")
    return articles

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Wikipedia-style text dataset")
    parser.add_argument("--count", type=int, default=1000, help="Number of articles to generate")
    parser.add_argument("--output", type=str, default="wikipedia_style_text.csv", help="Output CSV file path")
    parser.add_argument("--min_words", type=int, default=250, help="Minimum words per text")
    parser.add_argument("--max_words", type=int, default=800, help="Maximum words per text")
    
    args = parser.parse_args()
    
    # Generate and export articles
    generate_and_export_articles(
        count=args.count,
        min_words=args.min_words,
        max_words=args.max_words,
        output_file=args.output
    )


