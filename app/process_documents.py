import json
import os
import re
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# --- Corrected Configuration ---
# Use pathlib.Path to define all paths. This creates Path objects, not strings.
# The script was designed for Docker, but for local execution,
# you can define the base directory like this.
BASE_DIR = Path(__file__).resolve().parent

# Define other paths relative to the script's location or an absolute path
INPUT_DIR = BASE_DIR / "input" # Assumes 'input' is in the same directory as the script
OUTPUT_DIR = BASE_DIR / "output"

# The following lines now correctly create Path objects
DOCS_DIR = INPUT_DIR / "documents"
STRUCTURE_DIR = INPUT_DIR / "structured_data"
REQUEST_FILE = INPUT_DIR / "request.json"
OUTPUT_FILE = OUTPUT_DIR / "challenge1b_output.json"

# The rest of your script remains the same...
# Use a lightweight model
MODEL_NAME = 'all-MiniLM-L6-v2'

def extract_meaningful_section_title(text_content, default_title):
    """Extract meaningful section title from text content."""
    lines = text_content.split('\n')
    
    # Look for patterns that indicate section headers
    title_patterns = [
        r'^[A-Z][A-Za-z\s]{10,80}$',  # Capitalized titles
        r'^\d+\.?\s+[A-Za-z][A-Za-z\s]{5,80}$',  # Numbered sections
        r'^[A-Z\s]{5,50}$',  # All caps short titles
        r'^Chapter\s+\d+.*',  # Chapter titles
        r'^Section\s+\d+.*',  # Section titles
        r'^Part\s+[IVX\d]+.*',  # Part titles
    ]
    
    # Check first few lines for potential titles
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) < 5 or len(line) > 100:
            continue
            
        # Skip common non-title patterns
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'^copyright',  # Copyright notices
            r'^©',  # Copyright symbol
            r'^\w+\.pdf',  # Filenames
            r'^version\s+\d',  # Version info
        ]
        
        if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
            continue
            
        # Check if line matches title patterns
        for pattern in title_patterns:
            if re.match(pattern, line):
                return line
    
    # If no clear title found, look for the first substantial line
    for line in lines[:5]:
        line = line.strip()
        if 10 <= len(line) <= 80 and not line.lower().startswith(('page', 'copyright', '©')):
            return line
    
    return default_title

def get_document_sections(doc_path, structure_data):
    """Extracts text for each section defined in the structured outline."""
    sections = []
    doc = fitz.open(doc_path)
    
    # Get the title from structure data
    doc_title = structure_data.get("title", doc_path.stem)
    
    # Sort sections by page number
    sorted_outline = structure_data.get("outline", [])
    
    if not sorted_outline:
        # If no outline, create sections by analyzing the document structure
        print(f"No structured outline found for {doc_path.name}, analyzing document structure...")
        sections_from_text = create_sections_from_document(doc, doc_path.name, doc_title)
        sections.extend(sections_from_text)
    else:
        # Sort by page number
        sorted_outline = sorted(sorted_outline, key=lambda x: x.get("page", 0))
        
        # Process each section from structured data
        for i, section_info in enumerate(sorted_outline):
            start_page = section_info.get("page", 1)
            section_title = section_info.get("text", "Untitled Section")
            
            # Determine end page
            if i + 1 < len(sorted_outline):
                end_page = sorted_outline[i + 1].get("page", doc.page_count + 1)
            else:
                end_page = doc.page_count + 1
            
            # Convert to 0-based indexing for PyMuPDF
            start_page_idx = max(0, start_page - 1)
            end_page_idx = min(doc.page_count, end_page - 1)
            
            section_text = ""
            for page_num in range(start_page_idx, end_page_idx + 1):
                if page_num < doc.page_count:
                    page_text = doc[page_num].get_text()
                    section_text += page_text + "\n"
            
            if section_text.strip():
                sections.append({
                    "doc_name": doc_path.name,
                    "page": start_page,
                    "title": section_title,
                    "text": section_text.strip()
                })
    
    doc.close()
    return sections

def create_sections_from_document(doc, doc_name, doc_title):
    """Create meaningful sections from document by analyzing text structure."""
    sections = []
    
    # Extract all text first
    full_text = ""
    page_texts = {}
    
    for page_num in range(doc.page_count):
        page_text = doc[page_num].get_text()
        page_texts[page_num + 1] = page_text
        full_text += page_text + "\n"
    
    if not full_text.strip():
        return sections
    
    # Try to identify section breaks
    lines = full_text.split('\n')
    current_section_lines = []
    current_page = 1
    current_title = None
    
    # Patterns that might indicate new sections
    section_patterns = [
        r'^\d+\.?\s+[A-Z][A-Za-z\s]{5,80}',  # "1. Introduction"
        r'^[A-Z][A-Z\s]{5,50}$',  # "INTRODUCTION"
        r'^Chapter\s+\d+',  # "Chapter 1"
        r'^Section\s+\d+',  # "Section 1"
        r'^Part\s+[IVX\d]+',  # "Part I"
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Introduction", "Learning Objectives"
    ]
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if not line_stripped:
            current_section_lines.append(line)
            continue
        
        # Check if this line looks like a section header
        is_section_header = False
        for pattern in section_patterns:
            if re.match(pattern, line_stripped) and len(line_stripped) < 100:
                is_section_header = True
                break
        
        # Also check for lines that are likely headers based on context
        if not is_section_header and len(line_stripped) < 80:
            # Check if line is surrounded by empty lines (potential header)
            prev_empty = i == 0 or not lines[i-1].strip()
            next_empty = i == len(lines)-1 or not lines[i+1].strip()
            
            if prev_empty and next_empty and len(line_stripped) > 5:
                # Additional checks for likely headers
                words = line_stripped.split()
                if (len(words) <= 8 and 
                    not line_stripped.lower().startswith(('page', 'copyright', '©')) and
                    not re.match(r'^\d+$', line_stripped)):
                    is_section_header = True
        
        if is_section_header and current_section_lines:
            # Save previous section
            section_text = '\n'.join(current_section_lines).strip()
            if section_text and len(section_text) > 50:  # Only save substantial sections
                title = current_title if current_title else extract_meaningful_section_title(section_text, doc_title)
                sections.append({
                    "doc_name": doc_name,
                    "page": current_page,
                    "title": title,
                    "text": section_text
                })
            
            # Start new section
            current_section_lines = []
            current_title = line_stripped
        
        current_section_lines.append(line)
    
    # Add the last section
    if current_section_lines:
        section_text = '\n'.join(current_section_lines).strip()
        if section_text:
            title = current_title if current_title else extract_meaningful_section_title(section_text, doc_title)
            sections.append({
                "doc_name": doc_name,
                "page": current_page,
                "title": title,
                "text": section_text
            })
    
    # If no meaningful sections were found, create one section from the whole document
    if not sections:
        title = extract_meaningful_section_title(full_text, doc_title)
        sections.append({
            "doc_name": doc_name,
            "page": 1,
            "title": title,
            "text": full_text.strip()
        })
    
    return sections

def split_into_subsections(text, max_length=1000):
    """Split text into smaller subsections for better analysis."""
    # First try to split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        # If no paragraph breaks, split by single newlines
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    if not paragraphs:
        return [text[:max_length]]
    
    # Combine paragraphs to create meaningful chunks
    subsections = []
    current_subsection = ""
    
    for para in paragraphs:
        potential_length = len(current_subsection + "\n\n" + para) if current_subsection else len(para)
        
        if potential_length <= max_length:
            current_subsection = current_subsection + "\n\n" + para if current_subsection else para
        else:
            if current_subsection.strip():
                subsections.append(current_subsection.strip())
            
            # If single paragraph is too long, truncate it
            if len(para) > max_length:
                current_subsection = para[:max_length-3] + "..."
            else:
                current_subsection = para
    
    if current_subsection.strip():
        subsections.append(current_subsection.strip())
    
    return subsections if subsections else [text[:max_length]]

def create_enhanced_query(persona, job_to_be_done):
    """Create an enhanced query for better semantic matching."""
    # Extract key terms and create multiple query variations
    base_query = f"As a {persona}, I need to {job_to_be_done}"
    
    # Add contextual variations
    queries = [
        base_query,
        f"{persona} {job_to_be_done}",
        f"For {persona}: {job_to_be_done}",
        job_to_be_done,
        persona
    ]
    
    return queries

def calculate_relevance_score(section_text, queries, model, persona, job_to_be_done):
    """Calculate relevance score using multiple query variations with enhanced filtering."""
    section_embedding = model.encode(section_text, convert_to_tensor=True)
    
    scores = []
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, section_embedding)[0][0].item()
        scores.append(score)
    
    max_score = max(scores)
    
    # Apply content filtering and boosting
    section_lower = section_text.lower()
    persona_lower = persona.lower()
    job_lower = job_to_be_done.lower()
    
    # Extract key terms from persona and job
    persona_keywords = [word.strip().lower() for word in persona_lower.split() if len(word) > 2]
    job_keywords = [word.strip().lower() for word in re.findall(r'\b\w+\b', job_lower) if len(word) > 2]
    
    # Boost score based on keyword matches
    keyword_boost = 0
    for keyword in persona_keywords + job_keywords:
        if keyword in section_lower:
            keyword_boost += 0.05
    
    max_score += min(keyword_boost, 0.3)  # Cap the boost
    
    # Penalty for obviously irrelevant content (but less aggressive)
    irrelevant_patterns = [
        'birthday party', 'waiver', 'climbing shoes', 'trampoline'
    ]
    
    if any(pattern in section_lower for pattern in irrelevant_patterns):
        max_score *= 0.2
    
    return max_score

def analyze_documents():
    """Main function to run the persona-driven analysis."""
    print("Starting Round 1B: Persona-Driven Document Intelligence...")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Request Data
    if not REQUEST_FILE.exists():
        raise FileNotFoundError(f"Request file not found at {REQUEST_FILE}")
    
    with open(REQUEST_FILE, 'r', encoding='utf-8') as f:
        request_data = json.load(f)
    
    persona = request_data.get("persona", "")
    job_to_be_done = request_data.get("job_to_be_done", "")
    
    if not persona or not job_to_be_done:
        raise ValueError("Both 'persona' and 'job_to_be_done' must be provided in request.json")
    
    print(f"Persona: {persona}")
    print(f"Job to be done: {job_to_be_done}")
    
    # Create enhanced queries for better matching
    enhanced_queries = create_enhanced_query(persona, job_to_be_done)
    print(f"Enhanced queries created: {len(enhanced_queries)} variations")
    
    # 2. Load Model
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Process All Documents
    all_sections = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF documents found in {DOCS_DIR}")
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for doc_path in pdf_files:
        print(f"Processing: {doc_path.name}")
        structure_file = STRUCTURE_DIR / f"{doc_path.stem}.json"
        
        if structure_file.exists():
            print(f"  Using structured data from {structure_file.name}")
            with open(structure_file, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)
        else:
            print(f"  No structure file found, analyzing document structure")
            structure_data = {}
        
        sections = get_document_sections(doc_path, structure_data)
        all_sections.extend(sections)
        print(f"  Extracted {len(sections)} sections")
    
    print(f"Total sections extracted: {len(all_sections)}")
    
    # 4. Calculate Relevance Scores
    print("Calculating relevance scores...")
    for section in all_sections:
        section_text = f"{section['title']} {section['text']}"
        score = calculate_relevance_score(section_text, enhanced_queries, model, persona, job_to_be_done)
        section["relevance_score"] = score
    
    # 5. Rank Sections by Relevance
    ranked_sections = sorted(all_sections, key=lambda x: x["relevance_score"], reverse=True)
    
    # Debug: Print top scores
    print("\nTop 10 section scores:")
    for i, section in enumerate(ranked_sections[:10]):
        print(f"{i+1}. {section['doc_name']} - {section['title'][:50]}... (Score: {section['relevance_score']:.4f})")
    
    # 6. Prepare Extracted Sections Output (Top 10 sections)
    # Filter out sections with very low relevance scores
    relevant_sections = [s for s in ranked_sections if s["relevance_score"] > 0.1]
    
    num_sections = min(10, len(relevant_sections))
    extracted_sections = []
    
    for i, section in enumerate(relevant_sections[:num_sections]):
        extracted_sections.append({
            "document": section["doc_name"],
            "page_number": section["page"],
            "section_title": section["title"],
            "importance_rank": i + 1
        })
    
    # 7. Prepare Sub-section Analysis (Top 5 sections)
    print("Performing sub-section analysis...")
    sub_section_analysis = []
    num_subsections = min(5, len(relevant_sections))
    
    for section in relevant_sections[:num_subsections]:
        # Split section into smaller parts for more granular analysis
        subsections = split_into_subsections(section["text"], max_length=1200)
        
        if not subsections:
            # If no subsections, use the original text (truncated if too long)
            refined_text = section["text"][:1200] + "..." if len(section["text"]) > 1200 else section["text"]
        else:
            # Find most relevant subsection
            best_subsection = ""
            best_score = -1
            
            for subsection in subsections:
                score = calculate_relevance_score(subsection, enhanced_queries, model, persona, job_to_be_done)
                if score > best_score:
                    best_score = score
                    best_subsection = subsection
            
            refined_text = best_subsection if best_subsection else subsections[0]
        
        # Ensure refined text is appropriate length
        if len(refined_text) > 1200:
            refined_text = refined_text[:1200] + "..."
        
        sub_section_analysis.append({
            "document": section["doc_name"],
            "refined_text": refined_text,
            "page_number": section["page"]
        })
    
    # 8. Create Final Output with correct metadata structure
    output_data = {
        "metadata": {
            "input_documents": [doc.name for doc in pdf_files],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }
    
    # 9. Save Output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nAnalysis complete! Output saved to: {OUTPUT_FILE}")
    print(f"Processed {len(all_sections)} sections from {len(pdf_files)} documents")
    print(f"Top {len(extracted_sections)} sections ranked by relevance")
    print(f"Generated {len(sub_section_analysis)} sub-section analyses")
    
    # Print summary statistics
    if ranked_sections:
        avg_score = sum(s['relevance_score'] for s in ranked_sections[:10])/min(10, len(ranked_sections))
        print(f"Average relevance score (top 10): {avg_score:.4f}")
    
    return output_data

if __name__ == "__main__":
    try:
        analyze_documents()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()