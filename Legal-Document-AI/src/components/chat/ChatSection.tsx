import React, { useState, useRef, useEffect, lazy, Suspense } from "react";
import { Badge } from "@/components/ui/badge";
import { Send, Mic, Bot as BotIcon, User, Sidebar, Activity } from "lucide-react";
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
    // MODIFICATION: Removed 'mt-16 md:mt-0'
    // The parent 'main' element in ChatPage already handles the top offset
    // for the fixed mobile header.
    <div className="flex flex-col flex-1 min-h-0 bg-gradient-to-b from-black to-[#050506]">
      <style>{`
        .header { padding:12px 16px; display:flex; align-items:center; justify-content:space-between; background: linear-gradient(180deg,#061022,#091526); border-bottom:1px solid rgba(255,255,255,0.06); flex-shrink:0; }
        .header-left { display:flex; gap:12px; align-items:center; }
        .title { color:#f3f4f6; font-weight:600; font-size:18px; }
        .subtitle { color:#9ca3af; font-size:12px; margin-top:4px; }
        .robot-icon { width:56px; height:56px; border-radius:10px; display:grid; place-items:center; background:#0b1626; border:1px solid rgba(255,255,255,0.08); flex-shrink:0; }
        .messages { padding:20px; overflow-y:auto; flex:1; min-height:0; -webkit-overflow-scrolling: touch; }
        .msg-row { display:flex; gap:14px; align-items:flex-start; margin-bottom:12px; }
        .msg-row.user { flex-direction:row-reverse; }
        .avatar { width:36px; height:36px; border-radius:999px; display:grid; place-items:center; background:#0c0c0c; border:1px solid rgba(255,255,255,0.06); flex-shrink:0; }
        .bubble { border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px; background:#0b0b0b; max-width:800px; color:#e6e6e6; font-size:14px; word-break:break-word; }
        .typing-dots { display:flex; gap:6px; align-items:center; }
        .dot { width:6px; height:6px; background:#cfcfcf; border-radius:999px; opacity:0.3; animation: blink 1s infinite; }
        .dot:nth-child(2){ animation-delay:0.15s; }
        .dot:nth-child(3){ animation-delay:0.3s; }
        @keyframes blink { 0%,100%{opacity:0.3; transform:translateY(0);} 50%{opacity:1; transform:translateY(-4px);} }
        .sticky-input { flex-shrink:0; display:flex; justify-content:center; padding:12px; background:#000; border-top:1px solid rgba(255,255,255,0.06); }
        .form-wrap { width:100%; max-width:920px; display:flex; flex-direction:column; gap:6px; margin:0 auto; }
        .input-row { display:flex; gap:8px; align-items:center; padding:8px 12px; border-radius:999px; background:#111; border:1px solid rgba(255,255,255,0.08); }
        .input-row input { flex:1; background:transparent; border:0; outline:none; color:#e6e6e6; font-size:13px; padding:6px 0; }
        .suggest-row { display:flex; gap:10px; overflow-x:auto; padding:4px 2px 0; }
        .suggest-row::-webkit-scrollbar { height:6px; }
        .suggest-row::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.12); border-radius:999px; }
        .suggest-pill { flex:0 0 auto; white-space:nowrap; padding:8px 12px; border-radius:999px; background:#111; border:1px solid rgba(255,255,255,0.06); color:#e6e6e6; font-size:13px; cursor:pointer; }
        .desktop-toggles { display:inline-flex; gap:8px; align-items:center; }
        .desktop-toggle-btn { display:flex; gap:8px; align-items:center; padding:8px 10px; border-radius:999px; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); color:#e6e6e6; font-size:13px; cursor:pointer; }
        @media (max-width:900px) {
          .desktop-toggles { display:none; }
          .suggest-row { display:none; }
        }
      `}</style>

      {/* Header - This is 'flex-shrink: 0' from the CSS */}
      <div className="header" role="banner">
        <div className="header-left">
          <div className="robot-icon">
            <BotIcon size={28} color="#e6e6e6" />
          </div>
          <div className="min-w-0">
            <div className="title truncate">{titleDisplay}</div>
            {activeDocument && <div className="subtitle">{activeDocument}</div>}
          </div>
        </div>

        <div className="desktop-toggles">
          <div
            className="desktop-toggle-btn"
            onClick={() => setShowSourcesDesktop && setShowSourcesDesktop(!showSourcesDesktop)}
          >
            <Sidebar size={14} />
            <span>{showSourcesDesktop ? "Hide Sources" : "Show Sources"}</span>
          </div>
          <div
            className="desktop-toggle-btn"
            onClick={() => setShowInsightsDesktop && setShowInsightsDesktop(!showInsightsDesktop)}
          >
            <Activity size={14} />
            <span>{showInsightsDesktop ? "Hide Insights" : "Show Insights"}</span>
          </div>
        </div>
      </div>

      {/* Scrollable Messages - This is 'flex: 1' and 'overflow-y: auto' from the CSS */}
      <div className="messages flex-1" ref={containerRef}>
        {!hasDocuments && messages.length === 0 ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-full max-w-[700px] h-[300px] rounded-lg border border-white/10 overflow-hidden">
              <Suspense fallback={<div className="h-full grid place-items-center">Loading 3D…</div>}>
                <Spline scene="https://prod.spline.design/n1Lad8xaG0iocaRW/scene.splinecode" />
              </Suspense>
            </div>
            <h2 className="text-gray-200">Ask Anything</h2>
            <p className="text-gray-400 text-center max-w-[520px]">
              Upload your legal documents to begin AI-powered analysis and get instant insights.
            </p>
          </div>
        ) : (
          <>
            {messages.map((m) => (
              <div key={m.id} className={`msg-row ${m.sender === "user" ? "user" : ""}`}>
                <div className="avatar">
                  {m.sender === "ai" ? <BotIcon size={16} color="#9edbff" /> : <User size={16} color="#7efbb5" />}
                </div>
                <div className="bubble">
                  <div className="flex justify-between mb-1">
                    <Badge variant="outline" className="text-[11px] text-gray-300 border-white/10">
                      {m.sender === "ai" ? "AI Assistant" : "You"}
                    </Badge>
                    <span className="text-[11px] text-gray-500">
                      {m.timestamp instanceof Date ? m.timestamp.toLocaleTimeString() : new Date(m.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  {m.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="msg-row">
                <div className="avatar">
                  <BotIcon size={16} color="#9edbff" />
                </div>
                <div className="bubble">
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

      {/* Sticky Input - This is 'flex-shrink: 0' from the CSS */}
      <div className="sticky-input">
        <div className="form-wrap">
          <form onSubmit={handleSubmit} className="input-row">
            <input
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask about your documents..."
              disabled={isLoading}
            />
            <button type="button">
              <Mic size={14} color="#e6e6e6" />
            </button>
            <button type="submit" disabled={!inputText.trim() || isLoading}>
              <Send size={14} color="#fff" />
            </button>
          </form>

          <div className="suggest-row">
            {suggestedQuestions.map((q, i) => (
              <div key={i} className="suggest-pill" onClick={() => handleSuggested(q)}>
                {q}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};