import React, { useState, useRef, useEffect, lazy, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Send, Mic, Bot as BotIcon, User } from "lucide-react";
import { useChat } from "@/hooks/api";

const Spline = lazy(() => import("@splinetool/react-spline"));

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: Date;
  conversation_id?: string;
}

interface ChatSectionProps {
  activeDocument: string | null;
  hasDocuments: boolean;
  activeFileId?: string | null;
  onConversationIdChange?: (conversationId: string) => void; // Add this prop
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
  onConversationIdChange 
}: ChatSectionProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [visible, setVisible] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Use the real chat API
  const chatMutation = useChat();

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom(); }, [messages]);
  useEffect(() => {
    if (messages.length === 0) return;
    const last = messages[messages.length - 1];
    setTimeout(() => setVisible((prev) => [...prev, last.id]), 80);
  }, [messages]);

  // Update parent when conversation ID changes
  useEffect(() => {
    if (conversationId && onConversationIdChange) {
      onConversationIdChange(conversationId);
    }
  }, [conversationId, onConversationIdChange]);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;
    
    const userMessage: Message = { 
      id: Date.now().toString(), 
      text: text.trim(), 
      sender: "user", 
      timestamp: new Date(),
      conversation_id: conversationId || undefined
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInputText("");

    try {
      // Call the real API
      const response = await chatMutation.mutateAsync({
        message: text.trim(),
        file_id: activeFileId || undefined,
        conversation_id: conversationId || undefined,
      });

      // Update conversation ID if it's new
      if (!conversationId && response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      const aiMessage: Message = { 
        id: response.message_id || (Date.now() + 1).toString(), 
        text: response.response || "I received your message.", 
        sender: "ai", 
        timestamp: new Date(),
        conversation_id: response.conversation_id 
      };
      
      setMessages((prev) => [...prev, aiMessage]);

    } catch (error: any) {
      console.error('Chat error:', error);
      
      // Fallback error message
      const errorMessage: Message = { 
        id: (Date.now() + 1).toString(), 
        text: `Sorry, I encountered an error: ${error?.message || 'Please try again.'}`, 
        sender: "ai", 
        timestamp: new Date() 
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => { 
    e.preventDefault(); 
    sendMessage(inputText); 
  };

  const handleSuggested = (q: string) => sendMessage(q);
  const titleDisplay = activeDocument ? activeDocument.replace(/\.pdf$/i, "") : "Document Analysis";
  const isLoading = chatMutation.isPending;

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, background: "linear-gradient(180deg,#000,#050506)" }}>
      <style>{`
        .header { padding: 16px; margin: 16px; border-radius: 12px; display:flex; align-items:center; justify-content:space-between; background: linear-gradient(180deg,#061022,#091526); border:1px solid rgba(255,255,255,0.06); }
        .header-left { display:flex; gap:12px; align-items:center; }
        .title { color:#f3f4f6; font-weight:600; font-size:18px; }
        .subtitle { color:#9ca3af; font-size:12px; margin-top:4px; }
        .robot-icon { width:56px; height:56px; border-radius:10px; display:grid; place-items:center; background:#0b1626; border:1px solid rgba(255,255,255,0.08); }
        .chat-card { margin: 0 16px 24px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.005); flex:1; display:flex; flex-direction:column; }
        .messages { padding: 24px; overflow-y:auto; }
        .msg-row { display:flex; gap:14px; align-items:flex-start; margin-bottom:12px; }
        .msg-row.user { flex-direction:row-reverse; }
        .avatar { width:36px; height:36px; border-radius:999px; display:grid; place-items:center; background:#0c0c0c; border:1px solid rgba(255,255,255,0.06); }
        .bubble { border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px; background:#0b0b0b; max-width:800px; color:#e6e6e6; font-size:14px; }
        .typing-dots { display:flex; gap:6px; align-items:center; }
        .dot { width:6px; height:6px; background:#cfcfcf; border-radius:999px; opacity:0.3; animation: blink 1s infinite; }
        .dot:nth-child(2){ animation-delay:0.15s; }
        .dot:nth-child(3){ animation-delay:0.3s; }
        @keyframes blink { 0%,100%{opacity:0.3; transform:translateY(0);} 50%{opacity:1; transform:translateY(-4px);} }
        .floating-input { position:absolute; left:50%; transform:translateX(-50%); bottom:92px; width:calc(100% - 160px); max-width:760px; display:flex; gap:8px; align-items:center; padding:8px 12px; border-radius:999px; background:#111; border:1px solid rgba(255,255,255,0.08); box-shadow:0 6px 18px rgba(0,0,0,0.6); }
        .floating-input .input { flex:1; background:#111; border:none; color:#e6e6e6; font-size:13px; padding:6px 10px; outline:none; }
        .suggest-row { position:absolute; left:50%; transform:translateX(-50%); bottom:18px; width:calc(100% - 200px); max-width:740px; display:flex; gap:8px; overflow-x:auto; overflow-y:hidden; scrollbar-width:thin; scrollbar-color:rgba(255,255,255,0.1) transparent; }
        .suggest-row::-webkit-scrollbar { height:6px; }
        .suggest-row::-webkit-scrollbar-track { background:transparent; }
        .suggest-row::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:999px; }
        .suggest-pill { flex:0 0 auto; white-space:nowrap; padding:8px 12px; border-radius:999px; background:#111; border:1px solid rgba(255,255,255,0.08); color:#e6e6e6; font-size:12px; transition:all .14s ease; cursor:pointer; }
        .suggest-pill:hover { background:#1a1a1a; transform:translateY(-3px); }
        @media (max-width:900px){ .floating-input{ width:calc(100% - 48px); left:24px; transform:none; } .suggest-row{ width:calc(100% - 48px); left:24px; transform:none; } }
      `}</style>

      <div className="header">
        <div className="header-left">
          <div className="robot-icon">
            <BotIcon size={28} color="#e6e6e6" />
          </div>
          <div>
            <div className="title">{titleDisplay}</div>
            {activeDocument && <div className="subtitle">AI-powered legal document analysis</div>}
          </div>
        </div>
      </div>

      <div className="chat-card">
        <div className="messages" ref={containerRef}>
          {!hasDocuments && messages.length === 0 ? (
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:16 }}>
              <div style={{ width:"100%", maxWidth:700, height:340, borderRadius:12, border:"1px solid rgba(255,255,255,0.06)", overflow:"hidden" }}>
                <Suspense fallback={<div style={{height:"100%",display:"grid",placeItems:"center"}}>Loading 3D…</div>}>
                  <Spline scene="https://prod.spline.design/n1Lad8xaG0iocaRW/scene.splinecode" />
                </Suspense>
              </div>
              <h2 style={{ color:"#e6e6e6" }}>Ask Anything</h2>
              <p style={{ color:"#9ca3af", textAlign:"center", maxWidth:520 }}>Upload your legal documents to begin AI-powered analysis and get instant insights.</p>
            </div>
          ) : (
            <>
              {messages.map(m => (
                <div key={m.id} className={`msg-row ${m.sender==="user" ? "user":""}`}>
                  <div className="avatar">{m.sender==="ai"?<BotIcon size={16} color="#9edbff"/>:<User size={16} color="#7efbb5"/>}</div>
                  <div className="bubble">
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                      <Badge variant="outline" style={{ fontSize:11, color:"#ccc", borderColor:"rgba(255,255,255,0.08)" }}>
                        {m.sender==="ai"?"AI Assistant":"You"}
                      </Badge>
                      <span style={{ fontSize:11, color:"#777" }}>{m.timestamp.toLocaleTimeString()}</span>
                    </div>
                    {m.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="msg-row">
                  <div className="avatar"><BotIcon size={16} color="#9edbff" /></div>
                  <div className="bubble"><div className="typing-dots"><div className="dot"/><div className="dot"/><div className="dot"/></div></div>
                </div>
              )}
              <div ref={messagesEndRef}/>
            </>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="floating-input">
        <input 
          className="input" 
          value={inputText} 
          onChange={(e)=>setInputText(e.target.value)} 
          placeholder="Ask about your documents..." 
          disabled={isLoading}
        />
        <button 
          type="button" 
          style={{ width:36, height:36, borderRadius:8, background:"#1a1a1a", border:"1px solid rgba(255,255,255,0.08)" }}
        >
          <Mic size={14} color="#e6e6e6"/>
        </button>
        <button 
          type="submit" 
          disabled={!inputText.trim()||isLoading} 
          style={{ 
            width:36, 
            height:36, 
            borderRadius:8, 
            background: inputText.trim()&&!isLoading?"#0f86bf":"#1a1a1a", 
            border:"1px solid rgba(255,255,255,0.08)" 
          }}
        >
          <Send size={14} color="#fff"/>
        </button>
      </form>

      <div className="suggest-row">
        {suggestedQuestions.map((q,i)=>(
          <div key={i} className="suggest-pill" onClick={()=>handleSuggested(q)}>{q}</div>
        ))}
      </div>
    </div>
  );
};