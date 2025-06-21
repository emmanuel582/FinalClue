import google.generativeai as genai
import requests
import json
import re
import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass
class ForensicEvidence:
    victim_name: str
    age: int
    occupation: str
    location: str
    date_found: str
    time_found: str
    observations: List[str]
    toxicology: Dict[str, str]
    environmental_factors: Dict[str, str]
    physical_findings: List[str]

class MedicalKnowledgeBase:
    """Medical knowledge database for forensic analysis"""
    
    def __init__(self):
        self.drug_interactions = {
            "ethanol_diazepam": {
                "interaction_type": "synergistic_cns_depression",
                "severity": "potentially_fatal",
                "mechanism": "Enhanced GABA-mediated CNS depression",
                "symptoms": ["respiratory_depression", "cardiac_depression", "coma"],
                "lethal_dose_combination": "Much lower than individual lethal doses"
            }
        }
        
        self.toxicology_database = {
            "ethanol": {
                "therapeutic_range": "0-0.08 g/dL",
                "toxic_range": "0.15-0.30 g/dL",
                "lethal_range": ">0.30 g/dL",
                "elimination_rate": "0.015 g/dL/hour",
                "effects": {
                    "low": "mild_intoxication",
                    "moderate": "significant_impairment",
                    "high": "stupor_coma",
                    "lethal": "respiratory_cardiac_failure"
                }
            },
            "diazepam": {
                "therapeutic_range": "0.1-0.25 mg/L",
                "toxic_range": "0.5-2.0 mg/L",
                "lethal_range": ">2.0 mg/L",
                "half_life": "20-100 hours",
                "active_metabolites": ["desmethyldiazepam", "temazepam", "oxazepam"]
            }
        }
        
        self.postmortem_changes = {
            "rigor_mortis": {
                "onset": "2-6 hours",
                "peak": "12 hours",
                "resolution": "24-48 hours",
                "factors": ["temperature", "physical_activity", "cause_of_death"]
            },
            "body_temperature": {
                "normal_core": 37.0,
                "cooling_rate": "0.5-1.0°C/hour",
                "factors": ["ambient_temperature", "body_mass", "clothing", "air_circulation"]
            }
        }

class HuggingFaceAPI:
    """Interface for Hugging Face medical models"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def query_medical_model(self, text: str, model_name: str = "microsoft/DialoGPT-medium") -> Dict:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {"inputs": text}
        
        try:
            response = requests.post(f"{self.base_url}/{model_name}", 
                                   headers=headers, json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

class ForensicAnalyzer:
    """Main forensic analysis system"""
    
    def __init__(self, gemini_api_key: str, hf_token: str = None):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize knowledge bases
        self.medical_kb = MedicalKnowledgeBase()
        self.hf_api = HuggingFaceAPI(hf_token) if hf_token else None
        
    def parse_report(self, report_text: str) -> ForensicEvidence:
        """Extract structured data from forensic report"""
        
        prompt = f"""
        Extract and structure the following forensic report data into JSON format:
        
        {report_text}
        
        Return ONLY a JSON object with these exact keys:
        - victim_name
        - age
        - occupation
        - location
        - date_found
        - time_found
        - observations (array)
        - toxicology (object with substance names as keys)
        - environmental_factors (object)
        - physical_findings (array)
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean the response to extract JSON
            json_text = self._extract_json(response.text)
            data = json.loads(json_text)
            
            return ForensicEvidence(**data)
        except Exception as e:
            print(f"Error parsing report: {e}")
            return None
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from Gemini response"""
        # Find JSON in the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        return text
    
    def calculate_time_of_death(self, evidence: ForensicEvidence) -> Dict[str, str]:
        """Calculate estimated time of death using multiple methods"""
        
        # Extract numerical values
        core_temp = self._extract_temperature(evidence.environmental_factors.get("core_body_temperature", "29°C"))
        ambient_temp = self._extract_temperature(evidence.environmental_factors.get("room_temperature", "19°C"))
        
        # Henssge nomogram method (simplified)
        temp_difference = 37.0 - core_temp  # Normal body temp - current temp
        cooling_rate = 0.7  # Average cooling rate per hour
        estimated_hours = temp_difference / cooling_rate
        
        # Factor in ambient temperature
        if ambient_temp < 20:
            estimated_hours *= 0.8  # Faster cooling in cold environment
        
        # Cross-reference with rigor mortis
        rigor_analysis = self._analyze_rigor_mortis(evidence.physical_findings)
        
        time_found = datetime.datetime.strptime(f"{evidence.date_found} {evidence.time_found}", 
                                              "%B %d, %Y %I:%M %p")
        estimated_death_time = time_found - datetime.timedelta(hours=estimated_hours)
        
        return {
            "estimated_time_of_death": estimated_death_time.strftime("%B %d, %Y at %I:%M %p"),
            "method_used": "Combined thermometric and rigor mortis analysis",
            "confidence_interval": f"±2 hours",
            "supporting_evidence": rigor_analysis
        }
    
    def _extract_temperature(self, temp_str: str) -> float:
        """Extract temperature value from string"""
        match = re.search(r'(\d+\.?\d*)', temp_str)
        return float(match.group(1)) if match else 0.0
    
    def _analyze_rigor_mortis(self, physical_findings: List[str]) -> str:
        """Analyze rigor mortis findings"""
        rigor_mentions = [finding for finding in physical_findings 
                         if "rigor" in finding.lower()]
        
        if any("fully developed" in finding.lower() for finding in rigor_mentions):
            return "Rigor mortis fully developed - consistent with 8-12 hours post-mortem"
        elif any("partial" in finding.lower() for finding in rigor_mentions):
            return "Partial rigor mortis - consistent with 4-8 hours post-mortem"
        else:
            return "Rigor mortis status unclear from available evidence"
    
    def analyze_toxicology(self, evidence: ForensicEvidence) -> Dict[str, any]:
        """Comprehensive toxicological analysis"""
        
        analysis = {
            "substances_detected": list(evidence.toxicology.keys()),
            "interactions": [],
            "cause_of_death_assessment": "",
            "mechanism_of_death": ""
        }
        
        # Check for dangerous combinations
        substances = [s.lower() for s in evidence.toxicology.keys()]
        
        if "ethanol" in substances and "diazepam" in substances:
            interaction = self.medical_kb.drug_interactions["ethanol_diazepam"]
            analysis["interactions"].append({
                "combination": "Ethanol + Diazepam",
                "severity": interaction["severity"],
                "mechanism": interaction["mechanism"],
                "clinical_significance": "Synergistic CNS depression leading to respiratory failure"
            })
            
            analysis["cause_of_death_assessment"] = "Combined drug toxicity (ethanol and diazepam)"
            analysis["mechanism_of_death"] = "Respiratory and cardiac depression due to synergistic CNS depression"
        
        return analysis
    
    def generate_comprehensive_report(self, report_text: str) -> Dict[str, any]:
        """Generate complete forensic analysis report"""
        
        # Parse the input report
        evidence = self.parse_report(report_text)
        if not evidence:
            return {"error": "Could not parse forensic report"}
        
        # Perform analyses
        time_of_death = self.calculate_time_of_death(evidence)
        toxicology_analysis = self.analyze_toxicology(evidence)
        
        # Generate expert opinion using Gemini with enhanced medical knowledge
        expert_opinion_prompt = f"""
        You are a forensic pathologist with 20 years of experience. Based on the following evidence, 
        provide a professional forensic opinion:
        
        CASE SUMMARY:
        Victim: {evidence.victim_name}, {evidence.age} years old, {evidence.occupation}
        Found: {evidence.date_found} at {evidence.time_found}
        Location: {evidence.location}
        
        PHYSICAL EVIDENCE:
        {', '.join(evidence.physical_findings)}
        
        TOXICOLOGY:
        {json.dumps(evidence.toxicology, indent=2)}
        
        CALCULATED TIME OF DEATH:
        {time_of_death['estimated_time_of_death']}
        
        TOXICOLOGICAL ANALYSIS:
        {json.dumps(toxicology_analysis, indent=2)}
        
        Provide a detailed professional opinion including:
        1. Most probable cause of death
        2. Manner of death (natural, accident, suicide, homicide, undetermined)
        3. Contributing factors
        4. Confidence level in your assessment
        5. Any additional investigations recommended
        
        Write this as you would for a coroner's report or court testimony.
        """
        
        expert_response = self.model.generate_content(expert_opinion_prompt)
        
        # Compile final report
        final_report = {
            "case_information": {
                "victim": evidence.victim_name,
                "age": evidence.age,
                "occupation": evidence.occupation,
                "incident_location": evidence.location,
                "discovery_details": f"Found on {evidence.date_found} at {evidence.time_found}"
            },
            "estimated_time_of_death": time_of_death,
            "toxicological_findings": toxicology_analysis,
            "expert_forensic_opinion": expert_response.text,
            "report_generated": datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "analysis_methods": [
                "Thermometric time of death calculation",
                "Rigor mortis assessment",
                "Toxicological interaction analysis",
                "AI-enhanced medical knowledge correlation"
            ]
        }
        
        return final_report
    
    def save_report(self, report_data: Dict, filename: str = "forensic_analysis_report.json"):
        """Save the analysis report to file"""
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=4)
        print(f"Report saved to {filename}")

def main():
    """Main function to run forensic analysis"""
    
    # Initialize the analyzer
    GEMINI_API_KEY = "AIzaSyDadxiddfH31SyDxsNwZHd81YWbd7na1-k"  # Your API key
    
    try:
        analyzer = ForensicAnalyzer(GEMINI_API_KEY)
        
        # Read the report file
        with open("report.txt", "r") as f:
            report_content = f.read()
        
        print("🔬 Advanced Forensic Analysis System")
        print("=" * 50)
        print("Processing forensic evidence...")
        
        # Generate comprehensive analysis
        analysis_result = analyzer.generate_comprehensive_report(report_content)
        
        if "error" in analysis_result:
            print(f"❌ Error: {analysis_result['error']}")
            return
        
        # Display results
        print("\n📋 FORENSIC ANALYSIS REPORT")
        print("=" * 50)
        
        print(f"\n👤 CASE INFORMATION:")
        case_info = analysis_result["case_information"]
        for key, value in case_info.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n⏰ TIME OF DEATH ANALYSIS:")
        tod = analysis_result["estimated_time_of_death"]
        print(f"   Estimated Time: {tod['estimated_time_of_death']}")
        print(f"   Method: {tod['method_used']}")
        print(f"   Confidence: {tod['confidence_interval']}")
        
        print(f"\n🧪 TOXICOLOGICAL FINDINGS:")
        tox = analysis_result["toxicological_findings"]
        print(f"   Substances: {', '.join(tox['substances_detected'])}")
        print(f"   Cause Assessment: {tox['cause_of_death_assessment']}")
        print(f"   Mechanism: {tox['mechanism_of_death']}")
        
        print(f"\n👨‍⚕️ EXPERT FORENSIC OPINION:")
        print(analysis_result["expert_forensic_opinion"])
        
        # Save detailed report
        analyzer.save_report(analysis_result, "detailed_forensic_report.json")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Detailed report saved to: detailed_forensic_report.json")
        
    except FileNotFoundError:
        print("❌ Error: report.txt file not found in the current directory")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()