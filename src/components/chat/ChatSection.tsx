import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Send, Mic, Download, Bot, User } from "lucide-react";

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

interface ChatSectionProps {
  activeDocument: string | null;
  hasDocuments: boolean;
}

const suggestedQuestions = [
  "What are the key risks in this contract?",
  "Summarize this document in simple terms",
  "Are there any hidden fees or clauses?",
  "What are my obligations under this agreement?",
  "What happens if I terminate this contract?",
  "Are there any unusual or concerning terms?"
];

export const ChatSection = ({ activeDocument, hasDocuments }: ChatSectionProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `I've analyzed your question about "${text.trim()}". Based on the document analysis, here are the key points you should know...\n\nThis is a simulated response. In a real implementation, this would be powered by AI analysis of your uploaded documents.`,
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
    }, 2000);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputText);
  };

  const handleSuggestedQuestion = (question: string) => {
    sendMessage(question);
  };

  const exportConversation = () => {
    // Simulate PDF export
    console.log("Exporting conversation as PDF...");
  };

  return (
    <div className="flex-1 flex flex-col bg-background">
      {/* Header */}
      <div className="p-6 border-b border-border bg-card/50">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="heading-serif text-2xl">
              {activeDocument || "Document Analysis"}
            </h1>
            {hasDocuments && (
              <p className="text-sm text-muted-foreground mt-1">
                AI-powered legal document analysis
              </p>
            )}
          </div>
          
          {messages.length > 0 && (
            <Button 
              onClick={exportConversation}
              variant="outline" 
              size="sm"
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              Export PDF
            </Button>
          )}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6">
        {!hasDocuments ? (
          <div className="text-center py-12">
            <Bot className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="heading-serif text-xl mb-2">Upload Documents to Start</h3>
            <p className="text-muted-foreground">
              Upload your legal documents to begin AI-powered analysis and get instant insights.
            </p>
          </div>
        ) : messages.length === 0 ? (
          <div className="space-y-6">
            <div className="text-center py-8">
              <Bot className="h-12 w-12 mx-auto mb-4 text-primary" />
              <h3 className="heading-serif text-xl mb-2">Ready to Analyze</h3>
              <p className="text-muted-foreground">
                Ask me anything about your uploaded documents. I can explain complex terms, identify risks, and provide summaries.
              </p>
            </div>

            {/* Suggested Questions */}
            <div>
              <h4 className="text-sm font-medium mb-3 text-muted-foreground">Suggested Questions:</h4>
              <div className="grid gap-2">
                {suggestedQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => handleSuggestedQuestion(question)}
                    className="justify-start text-left h-auto py-3 px-4 hover:bg-primary/5 hover:border-primary/20"
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className="flex-shrink-0">
                  {message.sender === 'ai' ? (
                    <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-accent/10 rounded-full flex items-center justify-center">
                      <User className="h-4 w-4 text-accent" />
                    </div>
                  )}
                </div>
                
                <Card className={`p-4 max-w-2xl ${
                  message.sender === 'user' 
                    ? 'bg-primary/5 border-primary/20' 
                    : 'bg-card border-border'
                }`}>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Badge variant="outline" className="text-xs">
                      {message.sender === 'ai' ? 'AI Assistant' : 'You'}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.text}
                  </p>
                </Card>
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <Card className="p-4 bg-card border-border">
                  <Badge variant="outline" className="text-xs mb-2">
                    AI Assistant
                  </Badge>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-primary/60 rounded-full animate-pulse" />
                    <div className="w-2 h-2 bg-primary/60 rounded-full animate-pulse" style={{animationDelay: '0.2s'}} />
                    <div className="w-2 h-2 bg-primary/60 rounded-full animate-pulse" style={{animationDelay: '0.4s'}} />
                    <span className="ml-2 text-sm text-muted-foreground">AI is analyzing...</span>
                  </div>
                </Card>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      {hasDocuments && (
        <div className="p-6 border-t border-border bg-card/30">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask about your legal documents..."
                className="bg-background border-border"
                disabled={isTyping}
              />
            </div>
            <Button 
              type="button"
              size="icon"
              variant="outline"
              className="hover:bg-primary/10"
            >
              <Mic className="h-4 w-4" />
            </Button>
            <Button 
              type="submit"
              size="icon"
              disabled={!inputText.trim() || isTyping}
              className="bg-gradient-primary hover:shadow-glow-primary"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>
      )}
    </div>
  );
};