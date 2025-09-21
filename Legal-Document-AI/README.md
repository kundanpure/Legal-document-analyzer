# ⚖️ Legal Document AI – Frontend

This is the **frontend prototype** for our hackathon project:  
**Generative AI for Demystifying Legal Documents**  

It provides a modern, minimalistic Google-styled UI in dark theme with smooth animations, where users can upload legal documents, ask questions, and interact with AI-powered insights.

---

## ✨ Features

- **Landing Page**
  - Clean minimalistic design with parallax scrolling effects.
  - Hero section with *“Try Legal Document AI”* button.
  - Fully responsive across devices.

- **Chat Interface**
  - Three-panel layout:
    1. **Sources Panel** – Manage uploaded documents (limit: 50 PDFs).  
    2. **Chat Section** – Ask queries via text or voice. Suggested legal questions included.  
    3. **Insights Panel** – Access reports, summaries, and add notes.  
  - Progress bar animation with real-time backend process steps:
    - ✅ Document upload  
    - ✅ Text extraction via Document AI  
    - ✅ AI analysis via Gemini  
    - 🔄 Generating voice summary  
    - ⏳ Preparing interactive chat  

- **Reports & Visuals**
  - Risk scoring and document insights displayed with modern graphs.
  - Expandable AI insights panel.
  - Export reports and chat history (PDF).

- **UI/UX**
  - Built with **React + Tailwind CSS**.
  - Glassmorphism-style buttons and cards.
  - Dark theme with **Google Sans font**.
  - Smooth animations with **Framer Motion**.
  - Voice-enabled query input.

---

## 🛠️ Tech Stack

- **Frontend Framework:** React (Vite/CRA)
- **Styling:** Tailwind CSS + custom components
- **Animations:** Framer Motion
- **Icons:** Lucide React
- **Charts & Graphs:** Recharts
- **Font:** Google Sans
- **API Integration:** Axios → Python Flask Backend (connected with GCP services)

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- npm or yarn package manager
- Backend running (Flask API + GCP services)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/legal-docs-ai-frontend.git

# Navigate to project folder
cd legal-docs-ai-frontend

# Install dependencies
npm install
