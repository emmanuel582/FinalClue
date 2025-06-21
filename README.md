# 🔬 FinalClue - Advanced AI-Powered Forensic Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI](https://img.shields.io/badge/AI-Gemini%20%2B%20LangChain-orange.svg)](https://ai.google.dev/)

> **Professional-grade forensic analysis powered by cutting-edge AI technology**

## 🎯 Overview

FinalClue is a comprehensive forensic analysis system that combines advanced AI models (Google Gemini, LangChain) with extensive medical knowledge databases to provide professional-grade forensic pathology analysis. The system can analyze forensic evidence, calculate time of death, perform toxicological assessments, and generate detailed expert reports.

## ✨ Key Features

### 🔍 **Advanced Forensic Analysis**
- **Multi-Model AI Integration**: Combines Google Gemini 1.5 Flash with LangChain for enhanced analysis
- **Medical Knowledge Base**: Comprehensive database of drug interactions, toxicology data, and postmortem changes
- **Time of Death Calculation**: Uses multiple forensic methods including Henssge nomogram and rigor mortis analysis
- **Toxicological Assessment**: Advanced substance interaction analysis with postmortem considerations

### 📊 **Professional Reporting**
- **Structured Data Extraction**: AI-powered parsing of forensic reports into structured data
- **Comprehensive Reports**: Generate detailed forensic pathologist reports with expert opinions
- **Multiple Output Formats**: JSON and text report generation
- **Quality Assurance**: Built-in validation and cross-referencing with medical literature

### 🧠 **AI-Powered Intelligence**
- **Natural Language Processing**: Advanced text analysis and evidence interpretation
- **Medical Protocol Adherence**: Ensures compliance with standard forensic investigation protocols
- **Cross-Reference Analysis**: Validates findings against established medical knowledge
- **Confidence Scoring**: Provides reliability metrics for analysis results

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- (Optional) Hugging Face API token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/emmanuel582/FinalClue.git
cd FinalClue
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys**
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: Set Hugging Face token
export HF_TOKEN="your_huggingface_token_here"
```

### Basic Usage

#### 🔬 Basic Forensic Analysis
```python
from forensic import ForensicAnalyzer

# Initialize analyzer
analyzer = ForensicAnalyzer(gemini_api_key="your_key")

# Analyze forensic report
report_text = """
Victim: Dr. Daniel Eze
Age: 48
Occupation: Biochemistry Lecturer
Location: Staff quarters, University of Ibadan
Found: June 20, 2025 at 7:40 AM
Toxicology: ethanol 0.25 g/dL, diazepam 1.5 mg/L
Physical findings: Fully developed rigor mortis, cyanosis
"""

# Generate comprehensive analysis
results = analyzer.generate_comprehensive_report(report_text)
```

#### 🚀 Enhanced Analysis with LangChain
```python
from enhanced_forensic import AdvancedForensicAnalyzer

# Initialize enhanced analyzer
analyzer = AdvancedForensicAnalyzer(gemini_api_key="your_key")

# Perform advanced analysis
results = analyzer.generate_comprehensive_report(report_text)

# Save detailed reports
analyzer.save_detailed_report(results, "case_analysis")
```

## 📁 Project Structure

```
FinalClue/
├── forensic.py              # Basic forensic analysis system
├── enhanced_forensic.py     # Advanced analysis with LangChain
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── report.txt              # Sample forensic report
├── detailed_forensic_report.json  # Sample analysis output
└── forensic_analysis_*.json       # Generated analysis reports
```

## 🔧 Core Components

### 1. **ForensicAnalyzer** (`forensic.py`)
- **MedicalKnowledgeBase**: Comprehensive drug interaction and toxicology database
- **HuggingFaceAPI**: Interface for medical AI models
- **Time of Death Calculation**: Multi-method estimation using temperature and rigor mortis
- **Toxicological Analysis**: Substance interaction and lethal dose assessment

### 2. **AdvancedForensicAnalyzer** (`enhanced_forensic.py`)
- **LangChain Integration**: Advanced AI workflow management
- **Enhanced Medical Knowledge**: Extended database with forensic protocols
- **Quality Assurance**: Built-in validation and cross-referencing
- **Expert Opinion Generation**: Professional-grade forensic pathologist reports

## 📊 Analysis Capabilities

### 🕐 **Time of Death Estimation**
- **Henssge Nomogram Method**: Temperature-based calculation
- **Rigor Mortis Analysis**: Postmortem muscle stiffness assessment
- **Environmental Factors**: Ambient temperature and humidity considerations
- **Confidence Intervals**: Statistical reliability metrics

### 🧪 **Toxicological Analysis**
- **Substance Detection**: Identification of drugs and poisons
- **Interaction Assessment**: Synergistic and additive effects
- **Postmortem Considerations**: Redistribution and stability factors
- **Lethal Dose Evaluation**: Risk assessment and cause determination

### 📋 **Report Generation**
- **Structured Data Extraction**: AI-powered parsing of unstructured reports
- **Expert Opinion**: Professional forensic pathologist assessment
- **Multiple Formats**: JSON and human-readable text reports
- **Quality Scoring**: Analysis reliability metrics

## 🎯 Use Cases

### 🔍 **Forensic Pathology**
- Autopsy report analysis and interpretation
- Cause and manner of death determination
- Time of death estimation
- Toxicological evidence assessment

### 🏥 **Medical Examiner Offices**
- Standardized forensic analysis procedures
- Quality assurance and validation
- Expert opinion generation
- Case documentation and reporting

### 🎓 **Forensic Education**
- Training and educational purposes
- Case study analysis
- Forensic methodology demonstration
- Research and development

## 📈 Performance Metrics

- **Analysis Accuracy**: Validated against established forensic protocols
- **Processing Speed**: Real-time analysis with AI acceleration
- **Report Quality**: Professional-grade output suitable for legal proceedings
- **Scalability**: Handles multiple cases and complex evidence scenarios

## 🔒 Security & Privacy

- **API Key Management**: Secure handling of authentication credentials
- **Data Privacy**: No persistent storage of sensitive case information
- **Local Processing**: Option for offline analysis capabilities
- **Audit Trail**: Comprehensive logging for quality assurance

## 🤝 Contributing

We welcome contributions to improve FinalClue! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/emmanuel582/FinalClue.git
cd FinalClue

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI**: Advanced language model capabilities
- **LangChain**: AI workflow management and tool integration
- **Medical Community**: Forensic pathology expertise and validation
- **Open Source Community**: Contributing libraries and tools

## 📞 Support

For support, questions, or feature requests:
- 📧 Email: emmanuelwritecode@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/emmanuel582/FinalClue/issues)
- 📖 Documentation: [Wiki](https://github.com/emmanuel582/FinalClue/wiki)

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. All forensic analysis should be conducted by qualified professionals in accordance with local laws and regulations. The authors are not responsible for any misuse of this software.

---

**🔬 FinalClue - Where AI Meets Forensic Science** 🔬

*Empowering forensic professionals with cutting-edge AI technology*
