# Frontend Architecture: LegalMind AI

## Overview
LegalMind AI is a state-of-the-art intelligent document processing and chat platform tailored specifically for legal analysis. The frontend is built using React, TypeScript, and Vite, heavily leveraging modern design principles, including glassmorphism and fully responsive modular components.

## Technical Stack
- **Framework:** React 18
- **Build Tool:** Vite
- **Language:** TypeScript
- **Styling:** Tailwind CSS (with arbitrary value support and custom utility classes)
- **State Management:** React Query (TanStack Query v5) for server state; React Context API for authentication state
- **Authentication:** Firebase Auth (Google Provider)
- **Icons & Graphics:** Lucide React for iconography; Spline for 3D web-based rendering
- **Routing:** React Router DOM

## Core Features
- **Document Dashboard:** Displays previously uploaded files, insights, and analysis states. 
- **Upload Flow:** Multi-step modal using a drag-and-drop zone and seamless GCP Signed URL uploads.
- **Chat Interface:** Real-time conversational interface where users can question the Document AI regarding uploaded files. Supports context-awareness, source citations, and confidence metrics.
- **Insights Generation:** Generates concise text summaries, downloadable detailed PDF reports, and AI-synthesized audio summaries.
- **Authentication:** Split-screen authentication page featuring Google OAuth integration.

## Local Setup

### 1. Prerequisites
Ensure you have Node.js (v18+) and npm installed. 

### 2. Environment Variables
Create a `.env` file in the root directory by copying the `.env.example` file.
Configure the variables as follows:
- `VITE_API_URL`: The URL where the backend FastAPI server is running (typically http://localhost:8000).
- `VITE_FIREBASE_*`: Firebase SDK configuration credentials obtained from your Firebase Console.

### 3. Installation
Run the following commands to install dependencies and start the development server:
```bash
npm install
npm run dev
```

### 4. Build for Production
To build an optimized production bundle:
```bash
npm run build
npm run preview
```

## Application Structure
- `src/components/`: Reusable modular components such as navigation bars, modals, UI elements, and complex functional components.
  - `chat/`: Contains the conversational interface, chat bubbles, insights panel, and left sidebar.
  - `ui/`: Standardized UI elements (buttons, badges, modals, dialogs) usually generated from styling primitives.
- `src/pages/`: Main application views including Dashboard, Chat interface, Landing, and Authentication.
- `src/hooks/`: Custom React hooks, notably `api.ts` wrapping React Query for state persistence and server cache management.
- `src/lib/`: Core libraries and utilities, including the centralized API client and Firebase configuration.
- `src/contexts/`: React context providers for global state propagation (e.g., AuthContext).

## Best Practices & Guidelines
1. **API Integration:** All remote server calls must pass through `ApiService` defined in `src/lib/api.ts` to ensure consistent error handling, authorization header injection, and response parsing.
2. **State Caching:** Always use React Query for fetching asynchronous data to prevent redundant network requests.
3. **Styling Paradigm:** Strict adherence to Tailwind utility classes. Maintain the dark, glass-themed UI aesthetic by leveraging transparency and backdrop filters. Avoid standardizing on non-harmonious primary colors.

## Security Considerations
The frontend enforces user authentication via Firebase before allowing access to restricted components like the Dashboard and Chat interfaces. Cross-Origin Resource Sharing (CORS) configurations must align correctly with the backend to ensure secure API integration. 