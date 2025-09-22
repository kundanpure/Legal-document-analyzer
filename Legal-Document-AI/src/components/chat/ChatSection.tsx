import React, { useState, useRef, useEffect, lazy, Suspense } from "react";
import { Badge } from "@/components/ui/badge";
import { Send, Mic, Bot, User, Sidebar, Activity } from "lucide-react";
import { useChat } from "@/hooks/api";

const Spline = lazy(() => import("@splinetool/react-spline"));

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: Date | string;
  conversation_id?: string;
}

interface ChatSectionProps {
  activeDocument: string | null;
  hasDocuments: boolean;
  activeFileId?: string | null;
  onConversationIdChange?: (conversationId: string) => void;
  showSourcesDesktop?: boolean;
  setShowSourcesDesktop?: (v: boolean) => void;
  showInsightsDesktop?: boolean;
  setShowInsightsDesktop?: (v: boolean) => void;
  messages?: Message[];
  inputText?: string;
  setInputText?: (v: string) => void;
  sendMessage?: (text: string) => Promise<void>;
  isLoading?: boolean;
}

const suggestedQuestions = [
  "What are the key risks?",
  "Summarize this in simple terms",
  "Any hidden fees?",
  "What are my obligations?",
  "What if I terminate?",
  "Any unusual terms?",
];

export const ChatSection = ({
  activeDocument,
  hasDocuments,
  activeFileId,
  onConversationIdChange,
  showSourcesDesktop = true,
  setShowSourcesDesktop,
  showInsightsDesktop = true,
  setShowInsightsDesktop,
  messages: messagesProp,
  inputText: inputTextProp,
  setInputText: setInputTextProp,
  sendMessage: sendMessageProp,
  isLoading: isLoadingProp,
}: ChatSectionProps) => {
  const [internalMessages, setInternalMessages] = useState<Message[]>([]);
  const [internalInputText, setInternalInputText] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chatMutation = useChat();

  const messages = messagesProp ?? internalMessages;
  const inputText = inputTextProp ?? internalInputText;
  const setInputText = setInputTextProp ?? ((v: string) => setInternalInputText(v));
  const isLoading = typeof isLoadingProp === "boolean" ? isLoadingProp : chatMutation.isPending;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (conversationId && onConversationIdChange) onConversationIdChange(conversationId);
  }, [conversationId, onConversationIdChange]);

  const internalSendMessage = async (text: string) => {
    if (!text.trim()) return;
    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: "user",
      timestamp: new Date(),
      conversation_id: conversationId ?? undefined,
    };
    setInternalMessages((prev) => [...prev, userMessage]);
    setInternalInputText("");
    try {
      const response = await chatMutation.mutateAsync({
        message: text.trim(),
        file_id: activeFileId || undefined,
        conversation_id: conversationId || undefined,
      });
      if (!conversationId && response?.conversation_id) {
        setConversationId(response.conversation_id);
        if (onConversationIdChange) onConversationIdChange(response.conversation_id);
      }
      const aiMessage: Message = {
        id: response?.message_id || (Date.now() + 1).toString(),
        text: response?.response || "I received your message.",
        sender: "ai",
        timestamp: new Date(),
        conversation_id: response?.conversation_id,
      };
      setInternalMessages((prev) => [...prev, aiMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Sorry, I encountered an error: ${err?.message || "Please try again."}`,
        sender: "ai",
        timestamp: new Date(),
      };
      setInternalMessages((prev) => [...prev, errorMessage]);
    }
  };

  const sendMessage = sendMessageProp ?? internalSendMessage;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputText);
  };

  const handleSuggested = (q: string) => sendMessage(q);

  const titleDisplay = activeDocument ? activeDocument.replace(/\.pdf$/i, "") : "Document Analysis";

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-black to-[#050506] font-sans">
      <style>{`
        /* Header Styles */
        .chat-header { 
          padding: 16px 20px; 
          display: flex; 
          align-items: center; 
          justify-content: space-between; 
          background: linear-gradient(180deg, #061022, #091526); 
          border-bottom: 1px solid rgba(255,255,255,0.06); 
          flex-shrink: 0; 
          min-height: 72px;
        }
        
        .header-left { 
          display: flex; 
          gap: 16px; 
          align-items: center; 
          min-width: 0;
          flex: 1;
        }
        
        .robot-icon { 
          width: 48px; 
          height: 48px; 
          border-radius: 12px; 
          display: grid; 
          place-items: center; 
          background: #0b1626; 
          border: 1px solid rgba(255,255,255,0.08); 
          flex-shrink: 0; 
        }
        
        .title-section {
          min-width: 0;
          flex: 1;
        }
        
        .title { 
          color: #f3f4f6; 
          font-weight: 600; 
          font-size: 18px; 
          margin: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .subtitle { 
          color: #9ca3af; 
          font-size: 13px; 
          margin-top: 2px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        /* Messages Container */
        .messages-container { 
          flex: 1; 
          overflow-y: auto; 
          overflow-x: hidden;
          padding: 24px 20px;
          min-height: 0;
          -webkit-overflow-scrolling: touch;
        }
        
        .messages-container::-webkit-scrollbar {
          width: 6px;
        }
        .messages-container::-webkit-scrollbar-track {
          background: rgba(55, 65, 81, 0.1);
        }
        .messages-container::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.3);
          border-radius: 3px;
        }
        .messages-container::-webkit-scrollbar-thumb:hover {
          background: rgba(156, 163, 175, 0.5);
        }

        /* Empty State */
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          gap: 24px;
          text-align: center;
          padding: 40px 20px;
        }

        .spline-container {
          width: 100%;
          max-width: 500px;
          height: 250px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.1);
          overflow: hidden;
          background: rgba(11, 11, 11, 0.5);
        }

        /* Message Styles */
        .msg-row { 
          display: flex; 
          gap: 16px; 
          align-items: flex-start; 
          margin-bottom: 20px;
          max-width: 100%;
        }
        
        .msg-row.user { 
          flex-direction: row-reverse; 
        }
        
        .avatar { 
          width: 36px; 
          height: 36px; 
          border-radius: 50%; 
          display: grid; 
          place-items: center; 
          background: #0c0c0c; 
          border: 1px solid rgba(255,255,255,0.06); 
          flex-shrink: 0; 
        }
        
        .bubble { 
          border: 1px solid rgba(255,255,255,0.06); 
          border-radius: 16px; 
          padding: 16px; 
          background: #0b0b0b; 
          max-width: calc(100% - 60px);
          color: #e6e6e6; 
          font-size: 14px; 
          line-height: 1.5;
          word-wrap: break-word;
          overflow-wrap: break-word;
        }

        .message-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          gap: 12px;
        }

        .message-time {
          font-size: 11px;
          color: #6b7280;
          flex-shrink: 0;
        }

        /* Typing Animation */
        .typing-dots { 
          display: flex; 
          gap: 6px; 
          align-items: center; 
          padding: 8px 0;
        }
        
        .dot { 
          width: 6px; 
          height: 6px; 
          background: #9ca3af; 
          border-radius: 50%; 
          opacity: 0.3; 
          animation: blink 1.4s infinite; 
        }
        
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes blink { 
          0%, 100% { opacity: 0.3; transform: translateY(0); } 
          50% { opacity: 1; transform: translateY(-2px); } 
        }

        /* Input Section */
        .input-section { 
          flex-shrink: 0; 
          padding: 20px; 
          background: #000; 
          border-top: 1px solid rgba(255,255,255,0.06); 
        }
        
        .input-wrapper { 
          max-width: 800px; 
          margin: 0 auto;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .input-form { 
          display: flex; 
          gap: 12px; 
          align-items: center; 
          padding: 12px 16px; 
          border-radius: 24px; 
          background: #111; 
          border: 1px solid rgba(255,255,255,0.08); 
          transition: border-color 0.2s ease;
        }
        
        .input-form:focus-within {
          border-color: rgba(59, 130, 246, 0.3);
        }
        
        .chat-input { 
          flex: 1; 
          background: transparent; 
          border: 0; 
          outline: none; 
          color: #e6e6e6; 
          font-size: 14px; 
          padding: 4px 0;
          font-family: inherit;
        }
        
        .chat-input::placeholder {
          color: #6b7280;
        }
        
        .input-button {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 6px;
          background: transparent;
          border: none;
          cursor: pointer;
          border-radius: 8px;
          transition: background-color 0.2s ease;
        }
        
        .input-button:hover {
          background: rgba(255,255,255,0.05);
        }
        
        .input-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Suggestions */
        .suggestions { 
          display: flex; 
          gap: 8px; 
          overflow-x: auto; 
          padding: 4px 0;
          scrollbar-width: thin;
        }
        
        .suggestions::-webkit-scrollbar { 
          height: 6px; 
        }
        
        .suggestions::-webkit-scrollbar-thumb { 
          background: rgba(255,255,255,0.12); 
          border-radius: 3px; 
        }
        
        .suggestion-pill { 
          flex-shrink: 0; 
          white-space: nowrap; 
          padding: 8px 16px; 
          border-radius: 20px; 
          background: #111; 
          border: 1px solid rgba(255,255,255,0.06); 
          color: #e6e6e6; 
          font-size: 13px; 
          cursor: pointer; 
          transition: all 0.2s ease;
        }
        
        .suggestion-pill:hover {
          background: #1a1a1a;
          border-color: rgba(255,255,255,0.12);
        }

        /* Desktop Toggles */
        .desktop-toggles { 
          display: flex; 
          gap: 8px; 
          align-items: center; 
        }
        
        .toggle-button { 
          display: flex; 
          gap: 8px; 
          align-items: center; 
          padding: 8px 12px; 
          border-radius: 20px; 
          background: rgba(255,255,255,0.02); 
          border: 1px solid rgba(255,255,255,0.06); 
          color: #e6e6e6; 
          font-size: 13px; 
          cursor: pointer; 
          transition: all 0.2s ease;
          white-space: nowrap;
        }
        
        .toggle-button:hover {
          background: rgba(255,255,255,0.05);
          border-color: rgba(255,255,255,0.12);
        }

        /* Mobile Responsive */
        @media (max-width: 900px) {
          .desktop-toggles { 
            display: none; 
          }
          
          .suggestions { 
            display: none; 
          }
          
          .chat-header {
            padding: 12px 16px;
            min-height: 64px;
          }
          
          .header-left {
            gap: 12px;
          }
          
          .robot-icon {
            width: 40px;
            height: 40px;
          }
          
          .title {
            font-size: 16px;
          }
          
          .subtitle {
            font-size: 12px;
          }
          
          .messages-container {
            padding: 16px 16px;
          }
          
          .input-section {
            padding: 16px;
          }
          
          .bubble {
            max-width: calc(100% - 52px);
            padding: 12px;
          }
          
          .msg-row {
            gap: 12px;
            margin-bottom: 16px;
          }
          
          .avatar {
            width: 32px;
            height: 32px;
          }
          
          .spline-container {
            height: 200px;
          }
          
          .empty-state {
            padding: 20px 16px;
            gap: 20px;
          }
        }
      `}</style>

      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="robot-icon">
            <Bot size={24} color="#e6e6e6" />
          </div>
          <div className="title-section">
            <h1 className="title">{titleDisplay}</h1>
            {activeDocument && <div className="subtitle">{activeDocument}</div>}
          </div>
        </div>

        <div className="desktop-toggles">
          <button
            className="toggle-button"
            onClick={() => setShowSourcesDesktop && setShowSourcesDesktop(!showSourcesDesktop)}
          >
            <Sidebar size={14} />
            <span>{showSourcesDesktop ? "Hide Sources" : "Show Sources"}</span>
          </button>
          <button
            className="toggle-button"
            onClick={() => setShowInsightsDesktop && setShowInsightsDesktop(!showInsightsDesktop)}
          >
            <Activity size={14} />
            <span>{showInsightsDesktop ? "Hide Insights" : "Show Insights"}</span>
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div className="messages-container" ref={containerRef}>
        {!hasDocuments && messages.length === 0 ? (
          <div className="empty-state">
            <div className="spline-container">
              <Suspense fallback={
                <div className="h-full flex items-center justify-center text-gray-400">
                  Loading 3D Scene...
                </div>
              }>
                <Spline scene="https://prod.spline.design/n1Lad8xaG0iocaRW/scene.splinecode" />
              </Suspense>
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-gray-200 mb-3">Ask Anything</h2>
              <p className="text-gray-400 text-center max-w-md leading-relaxed">
                Upload your legal documents to begin AI-powered analysis and get instant insights.
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((m) => (
              <div key={m.id} className={`msg-row ${m.sender === "user" ? "user" : ""}`}>
                <div className="avatar">
                  {m.sender === "ai" ? 
                    <Bot size={16} color="#60a5fa" /> : 
                    <User size={16} color="#34d399" />
                  }
                </div>
                <div className="bubble">
                  <div className="message-header">
                    <Badge variant="outline" className="text-xs text-gray-300 border-gray-600 bg-gray-800/50">
                      {m.sender === "ai" ? "AI Assistant" : "You"}
                    </Badge>
                    <span className="message-time">
                      {m.timestamp instanceof Date ? 
                        m.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 
                        new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                      }
                    </span>
                  </div>
                  <div>{m.text}</div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="msg-row">
                <div className="avatar">
                  <Bot size={16} color="#60a5fa" />
                </div>
                <div className="bubble">
                  <div className="message-header">
                    <Badge variant="outline" className="text-xs text-gray-300 border-gray-600 bg-gray-800/50">
                      AI Assistant
                    </Badge>
                  </div>
                  <div className="typing-dots">
                    <div className="dot" />
                    <div className="dot" />
                    <div className="dot" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Section */}
      <div className="input-section">
        <div className="input-wrapper">
          <div onSubmit={handleSubmit} className="input-form">
            <input
              className="chat-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask about your documents..."
              disabled={isLoading}
            />
            <button type="button" className="input-button">
              <Mic size={16} color="#9ca3af" />
            </button>
            <button 
              type="submit" 
              disabled={!inputText.trim() || isLoading}
              className="input-button"
            >
              <Send size={16} color={!inputText.trim() || isLoading ? "#6b7280" : "#60a5fa"} />
            </button>
          </div>

          <div className="suggestions">
            {suggestedQuestions.map((q, i) => (
              <button 
                key={i} 
                className="suggestion-pill" 
                onClick={() => handleSuggested(q)}
                disabled={isLoading}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};