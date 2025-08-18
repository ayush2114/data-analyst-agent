---
title: Data Analyst Agent
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 📊 TDS Data Analyst Agent

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-Gemini%202.5-4285F4)

A powerful web application that combines Large Language Models with data analysis capabilities, allowing users to upload their datasets and ask questions in natural language.

![Screenshot](https://i.imgur.com/placeholder.png)

## ✨ Features

- **🤖 AI-Powered Analysis**: Uses Google's Gemini 2.5 Pro LLM to interpret questions and generate data insights
- **📈 Automatic Visualization**: Creates beautiful charts and graphs based on your data
- **🌐 Web Scraping**: Can fetch and analyze data from websites when no dataset is provided
- **📂 Multiple File Formats**: Supports CSV, Excel, JSON, Parquet and more
- **💪 Robust Architecture**: Built with failover mechanisms and error handling
- **🚀 Simple Interface**: Just upload your questions and data, then get answers

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip package manager
- Google Gemini API key(s)

### Installation

1. Clone the repository
```bash
git clone https://github.com/ayush2114/data-analyst-agent.git
cd data-analyst-agent
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys
```bash
cp .sample-env .env
# Edit the .env file and add your Gemini API keys
```

4. Run the application
```bash
./serve.sh
# Or directly with: uvicorn app:app --host 0.0.0.0 --port 8000
```

5. Visit `http://localhost:8000` in your browser

## 📝 How to Use

1. **Prepare Your Questions**
   - Create a text file with your questions
   - You can format specialized keys with: `- `key_name`: type`

2. **Prepare Your Data (Optional)**
   - You can upload a dataset file (CSV, Excel, etc.)
   - If no data is provided, the agent will try to find relevant data online

3. **Upload and Analyze**
   - Upload both files through the web interface
   - Click "Analyze Data" and wait for results
   - View visualizations and answers

## 🛠️ Technologies

- **FastAPI**: High-performance web framework
- **Langchain**: For LLM orchestration and agents
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Google Gemini 2.5**: State-of-the-art LLM
- **Docker**: Containerization

## 🔒 Security Notes

- API keys are stored locally in your `.env` file
- Data is processed locally and not stored permanently
- LLM requests are sent to Google's Gemini API

## 🌟 Example Use Cases

- **Market Research**: "Analyze market trends for product X over the last 5 years"
- **Financial Analysis**: "Calculate the ROI for these investments and visualize them"
- **Scientific Research**: "Find correlations between variables in my experiment data"
- **Business Intelligence**: "Summarize monthly sales performance by region"

## 🔄 Deployment

This application includes GitHub Actions workflows to automatically deploy to Hugging Face Spaces:

```yaml
name: Sync to Hugging Face
on:
  push:
    branches: [ main ]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Push to Hugging Face
        run: |
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git config --global user.name "github-actions[bot]"
          git remote add hf https://USER:${{ secrets.HF_TOKEN }}@huggingface.co/spaces/YOUR_SPACE/NAME
          git push hf main --force
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Google for providing the Gemini API
- The LangChain team for their excellent framework
- All open-source contributors to the libraries used in this project

---
