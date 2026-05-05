"""
Simplified PDF Annotation System
Just processes all papers with all questions using prompt caching
"""

import anthropic
import json
import os
from pathlib import Path
import base64
from datetime import datetime

API_KEY = 'ASU_TOKEN'

class PaperAnnotator:
    def __init__(self, api_key=API_KEY):
        """Initialize with Anthropic API key"""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Set it or pass it to __init__")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def load_pdf_as_base64(self, pdf_path):
        """Load PDF file and encode as base64"""
        with open(pdf_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    
    def build_system_prompt(self, few_shot_examples=None):
        """Build system prompt with optional few-shot examples"""
        parts = [
            "You are an human-centered computer scientist analyzing research papers.",
            "Please read this paper and answer questions about it. Only extract information from the paper, do not make anything up.",
            "Please only put the exact anster to the question in the 'Answer field.' If you have any reasoning for your answer, place it in the 'Reasoning' field.",
            "Please display how confident you are in each answer in the corresponding 'Confidence' field, giving a score between 0.0 and 1.0, where 0.0 means you are not confident at all, and 1.0 means you are completely condfident."
        ]
        
        if few_shot_examples:
            parts.append("\nExamples of correct annotations:\n")
            for i, ex in enumerate(few_shot_examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Q: {ex['question']}")
                parts.append(f"A: {ex['answer']}")
                if ex.get('reasoning'):
                    parts.append(f"Reasoning: {ex['reasoning']}")
                parts.append("")
        
        parts.append("\nResponse format:")
        parts.append("Answer: <your answer>")
        parts.append("Reasoning: <explanation>")
        parts.append("Confidence: <0.0-1.0>")
        
        return "\n".join(parts)
    
    def parse_response(self, response_text, question, paper_name=""):
        """Parse Claude's response into structured annotation"""
        import re
        
        answer_match = re.search(r"(?i)^Answer:\s*(.*?)(?=\n(?:Reasoning:|Confidence:|$))", response_text, re.MULTILINE | re.DOTALL)
        reasoning_match = re.search(r"(?i)^Reasoning:\s*(.*?)(?=\n(?:Confidence:|Answer:|$))", response_text, re.MULTILINE | re.DOTALL)
        confidence_match = re.search(r"(?i)^Confidence:\s*([\d.]+)", response_text, re.MULTILINE)
        
        # Get answer and clean it up
        answer = answer_match.group(1).strip() if answer_match else "Unknown"
        
        # For multi-select questions, convert to list
        if "select all that apply" in question.lower():
            # Split by commas, semicolons, or newlines and clean
            answer = [item.strip() for item in re.split(r'[,;\n]', answer) if item.strip()]
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."
        
        return {
            "paper": paper_name,
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.0
        }
    
    def annotate_paper(self, pdf_path, questions, few_shot_examples=None):
        """
        Annotate a single paper with all questions using prompt caching
        """
        print(f"\n{'='*70}")
        print(f"📄 Processing: {Path(pdf_path).name}")
        print(f"{'='*70}")
        
        pdf_base64 = self.load_pdf_as_base64(pdf_path)
        system_prompt = self.build_system_prompt(few_shot_examples)
        
        annotations = []
        cache_hits = 0
        errors = 0
        
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question[:60]}...", end=" ")
            
            try:
                message = self.client.messages.create(
                    model="claude-opus-4-1-20250805",
                    max_tokens=2000,
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                },
                                "cache_control": {"type": "ephemeral"}
                            },
                            {"type": "text", "text": question}
                        ]
                    }]
                )
                
                # Track caching
                if hasattr(message.usage, 'cache_read_input_tokens') and message.usage.cache_read_input_tokens:
                    cache_hits += 1
                    print("✓ (cached)")
                else:
                    print("✓")
                
                response_text = message.content[0].text
                annotation = self.parse_response(response_text, question, Path(pdf_path).name)
                annotations.append(annotation)
                
            except Exception as e:
                errors += 1
                print(f"✗ Error: {e}")
                annotations.append({
                    "paper": Path(pdf_path).name,
                    "question": question,
                    "answer": "ERROR",
                    "reasoning": str(e),
                    "confidence": 0.0
                })
        
        print(f"\n  ✅ Complete: {len(questions) - errors}/{len(questions)} successful")
        print(f"  💰 Cache efficiency: {cache_hits}/{len(questions)} questions used cache")
        
        return annotations
    
    def annotate_all_papers(self, pdf_directory, questions, few_shot_examples=None, output_dir="outputs"):
        """
        Process all papers in a directory
        
        Args:
            pdf_directory: Path to directory containing PDFs
            questions: List of questions to ask
            few_shot_examples: Optional examples for few-shot learning
            output_dir: Where to save results
        """
        print("\n" + "="*70)
        print("PAPER ANNOTATION - BATCH PROCESSING")
        print("="*70)
        
        
        # Convert pdf_directory to Path object
        pdf_dir = Path(pdf_directory)
        
        # Get all PDF files
        pdf_filenames = ["butler19.pdf", "black11.pdf", "keate94.pdf", "jack00.pdf",
        "brashear06.pdf", "kaminer14.pdf", "mohammed06.pdf", "kuwahara06.pdf", "jiang23.pdf",
        "gadiraju19.pdf", "madeo11.pdf", "luna23.pdf", "anderson96.pdf", "india19.pdf",
        "hossain22.pdf", "illijima22.pdf", "galbraith14.pdf", "boyd17.pdf", "karpodini22.pdf",
        "gennari08.pdf", "hornof03.pdf", "li22.pdf", "gotfrid16.pdf", "husaan11.pdf",
        "aruanno18.pdf", "bondioli17.pdf", "chibaudel20.pdf", "hurd19.pdf", "langford21.pdf",
        "gadiraju21.pdf", "gay20.pdf", "lu22.pdf", "biemanns09.pdf", "gleason20.pdf",
        "baloian02.pdf", "madjaroff17.pdf", "brule16.pdf", "ahmetovic21.pdf", "jiang22.pdf",
        "kamel02.pdf", "mok22.pdf", "alankus11.pdf", "hossain23.pdf", "jayant11.pdf",
        "kwon19.pdf", "alsaleem19.pdf", "mahmud23.pdf", "collins23.pdf", "lehman98.pdf",
        "carrington17.pdf", "lobo21.pdf", "milne13.pdf", "hoffmann15.pdf", "fell03.pdf",
        "gerling13.pdf", "khurana21.pdf"]
        
        # Create full paths by combining directory + filename
        pdf_files = [pdf_dir / filename for filename in pdf_filenames]

        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            return
        
        print(f"\n📊 Processing Summary:")
        print(f"  • Papers found: {len(pdf_files)}")
        print(f"  • Questions per paper: {len(questions)}")
        print(f"  • Total annotations: {len(pdf_files) * len(questions)}")
        print(f"  • Output directory: {output_dir}")
        print(f"  • Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each paper
        all_results = []
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'='*70}")
            print(f"Paper {i}/{len(pdf_files)}")
            print(f"{'='*70}")
            
            try:
                annotations = self.annotate_paper(
                    str(pdf_path),
                    questions,
                    few_shot_examples
                )
                
                # Save individual paper results
                output_file = Path(output_dir) / f"annotations_{pdf_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)
                
                all_results.append({
                    "paper": pdf_path.name,
                    "annotations": annotations
                })
                
                successful += 1
                print(f"  💾 Saved to: {output_file}")
                
            except Exception as e:
                failed += 1
                print(f"  ❌ Fatal error processing {pdf_path.name}: {e}")
                continue
        
        # Save combined results
        combined_output = Path(output_dir) / "all_annotations.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"✅ Successful: {successful}/{len(pdf_files)} papers")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(pdf_files)} papers")
        print(f"📁 Individual results: {output_dir}/annotations_*.json")
        print(f"📁 Combined results: {combined_output}")
        print(f"⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        return all_results


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Your questions
    all_questions = [
        "What is the DOI for this paper?",
        "What is the title of the paper?",
        "Who is the lead author of the paper?",
        "What year was this paper published?",
        "How many pages long is this paper?",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) totally fun first (i.e., Entertainment game reappropriated), 
        (2) mostly fun first (i.e., design workshop with disabled stakeholders),
        (3) both fun and utility first (i.e., design workshop with diverse stakeholders), 
        (4) mostly utility first (i.e., primary utilitarian mechanic designed first),
        (5) totally utility first (i.e., made by clinicians or specialists, fun incorporated later)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) unstructured play (i.e., no rules, making art),
        (2) semi-structured play (i.e., playground activities),
        (3) flexible structure with rules (i.e., improv),
        (4) flexible game (i.e., board game with house rules),
        (5) game with rigid rules (i.e., video game)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely skill-based (i.e., trivia, sports),
        (2) mostly skill-based (i.e., mario kart),
        (3) equally skill and chance-based (i.e., catan),
        (4) mostly chance-based (i.e., Uno),
        (5) entirely chance-based (i.e., all dice)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely solo (i.e., solitaire),
        (2) mostly solo (i.e., playing against AI in single-player StarCraft),
        (3) mix of solo and social affordances (i.e., animal crossing),
        (4) mostly social, but aspects are independent (i.e., house on hill haunt transition),
        (5) entirely social (i.e., tag)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely turn-based (i.e., tic tac toe),
        (2) follows a set of steps (i.e., viticulture's dynamic turn ordering),
        (3) turns are taken, but some actions can be taken at any time (i.e., pandemic),
        (4) most actions can be taken at any time, but there are some phases (i.e., PvP death reset timer),
        (5) entirely simultaneous (i.e., race)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely synchronous (i.e., real time strategy game),
        (2) mostly synchronous with a few asynchronous affordances (i.e., league of legends),
        (3) equal mix of synchronous and asynchronous affordances (i.e., helldivers 2),
        (4) mostly asynchronous with few synchronous affordances (i.e. cookie clicker),
        (5) entirely asynchronous (i.e., chess by postage mail)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely competitive (i.e., spit card game),
        (2) mostly competitive, but sometimes collaboration is important (i.e., forbidden island),
        (3) mix of competitive and collaborative (i.e., mario party 2 vs 2 mini-games),
        (4) mostly collaborative (i.e., animal crossing),
        (5) entirely collaborative (i.e., pandemic)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely symmetrical (i.e., tic tac toe),
        (2) mostly symmetrical (i.e., undertale's differing NPC behaviors based on previous playthroughs),
        (3) both symmetrical and asymmetrical (i.e., pandemic with its similar turn structure but different character abilities),
        (4) mostly asymmetrical (i.e., tag),
        (5) entirely asymmetrical (i.e., keep talking, nobody explodes)""",
        
        """How would you classify the experiential play value? Do not add additional context or reasoning in your response: 
        sensority (i.e. kaleidoscope, experiencing art), 
        fantasy (i.e. role-playing), 
        construction (i.e. music, painting, building), 
        challenge (testing physical or mental abilities against others or self), 
        undisclosed/unknown, but the paper does include a game/play system or experience, 
        not applicable""",
        
        """Which of the following study methods applies to this paper? Please select all that apply:
        workshop or design session, field study, usability testing, case study, focus group,
        controlled experiment, survey, telemetry/big data/cscw, secondary analysis, no data collected,
        other (please specify)""",
        
        """Which of the following interview methodologies was used? Select all that apply: 
        structured interview, semi-structured interview, contextual inquiry, not applicable, 
        other (please specify)""",
        
        """Which of the following workshop methodologies were used? Select all that apply:
        action research, cooperative method development, speculative design, persona, scenario, role playing, 
        affinity diagram, ideation, user journey, brainstorming, bodystorming, design probe, prototyping, mock-up,
        sketching, wireframing, card sorting, storyboarding, use case theater, object theater, not applicable,
        other (please specify)""",
        
        """Which of the following field study methodologies were used? Please select all that apply: autoethnography,
        ethnography, diary study, cultural, Wizard of Oz, not applicable, other (please specify)""",
        
        """Which of the following usability methodologies were used? Please select all that apply: expert analysis, think 
        aloud, cognitive walkthrough, heuristic analysis, not applicable, other (please specify)""",
        
        """Which of the following technology modalities were used? Please select all that apply: mobile, tablet, wearable, IoT, 
        assistive devices, robot, tangible interface, PC, virtual reality, augmented reality, game console, no technology, other (please specify)""",
        
        """What was the context of the study? Please select all that apply: clinic, public space (i.e. bowling alley), home, school, research lab, 
        social media, disability community space (i.e. Day program), remote/Zoom, not applicable, other (please specify)""",
        
        """What was the community of focus? Please select all that apply: Blind or low vision (BLV), Deaf or hard of hearing (DHH), Autism, 
        intellectual or developmental disability (IDD), motor or physical impairment, communication/speech, cognitive impairment, older adult, 
        general disability or accessibility, other (please specify)""",
        
        """What were the participant groups included in the study? Please select all that apply: People with disabilities, older adults, caregivers, 
        specialists (e.g. therapists, teachers), people without disabilities, no user involvement, other (please specify)""",
        
        """Please select the option(s) that best describe user involvement in the study: participatory design with stakeholders without disabilities, 
        participatory design with stakeholders with disabilities, user evaluation with stakeholders without disabilities, user evaluation with stakeholders 
        with disabilities, no representative user involvement, not applicable""",
        
        """Which methods of participant recruitment were used? Please select all that apply: phone, mail, email, convenience sampling (i.e. Day program), 
        snowball, word of mouth, flier, social media, clinic, no user involvement, undisclosed, other (please specify)""",
        
        """Which of the following issues were addressed in the study? Please select all that apply: increasing independence, increasing digital access, 
        increasing physical access, increasing understanding of users, supporting communication, personal informatics and changing behavior,
        education, increasing opportunities for enrichment, other""",
        
        """What is the type of contribution the study makes? Please select all that apply: empirical, artifact, methodological, theoretical, dataset, survey"""
    ]
    
    # Optional: Add few-shot examples here
    few_shot_examples = None
    # few_shot_examples = [
    #     {
    #         "question": "What is the title of the paper?",
    #         "answer": "Designing Accessible Games",
    #         "reasoning": "Found in the paper header and abstract"
    #     }
    # ]
    
    # Configuration
    PAPERS_DIR = "../papers"  # Change this to your papers directory
    OUTPUT_DIR = "outputs/opus"    # Change this to your desired output directory
    
    # Initialize annotator
    annotator = PaperAnnotator()
    
    # Process all papers
    results = annotator.annotate_all_papers(
        pdf_directory=PAPERS_DIR,
        questions=all_questions,
        few_shot_examples=few_shot_examples,
        output_dir=OUTPUT_DIR
    )