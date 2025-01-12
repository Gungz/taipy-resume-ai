# Resume AI with Taipy and Gemini

An AI-powered resume analyzer that provides insights, grammar checks, and interactive chat capabilities using Google's Gemini model.

## Features

- 📊 Resume Analysis Dashboard
- 🔍 Grammar Check & Suggestions
- 💬 Interactive Chat with Resume Context
- 📈 Visual Metrics & Word Cloud
- 📱 Responsive UI

## Prerequisites

- Python 3.10+
- Google API Key
- PyMuPDF (for PDF processing)
- Taipy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables: Create a .env file in the root directory:
```markdown
GOOGLE_API_KEY=your_google_api_key_here
```
Get your Google API key from [here](https://aistudio.google.com/app/apikey)

This project uses *gemini-1.5-flash*

## Local Deployment
1. Start the development server:
```bash
taipy run main.py --use-reloader
```

2. Acccess the application at http://localhost:10000


## Environment Variables
GOOGLE_API_KEY: Your Google API key for Gemini

## Sample Questions for AI
1. What do you think of my resume?
2. Do you think I can apply to *different job domain than currently* ? E.g. you are IT, but you want to apply to a job in HR.
3. What should I show in my resumre for career switch?
4. Do you think I should try applying to *higher position* ? E.g. you are a junior, but you want to apply to a senior position or a manager position.
5. What should I do to improve my resume?
6. I want a fresh start in my career, can you help me rewrite my resume?


## Contributing
1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request


## Possible Improvements
1. Resume generation or makeover from existing and download
2. Resume analysis for multiple resumes at once
3. Authentication and membership system