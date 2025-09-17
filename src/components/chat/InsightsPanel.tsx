import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { FileText, Volume2, BarChart3, StickyNote, Download, Play } from "lucide-react";

interface InsightsPanelProps {
  activeDocument: string | null;
  hasDocuments: boolean;
}

export const InsightsPanel = ({ activeDocument, hasDocuments }: InsightsPanelProps) => {
  const [userNotes, setUserNotes] = useState("");
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const generateSummary = async () => {
    setIsGeneratingSummary(true);
    setTimeout(() => setIsGeneratingSummary(false), 2000);
  };

  const generateAudioOverview = async () => {
    setIsGeneratingAudio(true);
    setTimeout(() => setIsGeneratingAudio(false), 3000);
  };

  const generateDetailedReport = async () => {
    setIsGeneratingReport(true);
    setTimeout(() => setIsGeneratingReport(false), 4000);
  };

  return (
    <div
      className="w-80 flex flex-col h-full text-gray-100 border-l"
      style={{
        background: "linear-gradient(180deg,#000,#050505)",
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <style>{`
        .ins-card {
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 12px;
          background: rgba(255,255,255,0.02);
          transition: all 0.25s ease;
        }
        .ins-card:hover {
          transform: translateY(-4px);
        }

        /* Accent Themes */
        .accent-green { color: #34d399; }
        .accent-purple { color: #818cf8; }
        .accent-red { color: #f87171; }
        .accent-amber { color: #fbbf24; }

        .btn-green {
          background: rgba(52,211,153,0.12);
          border: 1px solid rgba(52,211,153,0.25);
          color: #34d399;
        }
        .btn-green:hover { background: rgba(52,211,153,0.18); }

        .btn-purple {
          background: rgba(129,140,248,0.12);
          border: 1px solid rgba(129,140,248,0.25);
          color: #818cf8;
        }
        .btn-purple:hover { background: rgba(129,140,248,0.18); }

        .btn-red {
          background: rgba(248,113,113,0.12);
          border: 1px solid rgba(248,113,113,0.25);
          color: #f87171;
        }
        .btn-red:hover { background: rgba(248,113,113,0.18); }

        .btn-amber {
          background: rgba(251,191,36,0.12);
          border: 1px solid rgba(251,191,36,0.25);
          color: #fbbf24;
        }
        .btn-amber:hover { background: rgba(251,191,36,0.18); }
      `}</style>

      {/* Header */}
      <div className="p-6 border-b border-[rgba(255,255,255,0.08)]">
        <h2 className="heading-sans text-xl font-medium">Insights</h2>
        {activeDocument && (
          <p className="text-xs text-gray-400 mt-1 truncate">{activeDocument}</p>
        )}
      </div>

      {!hasDocuments ? (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="text-center text-gray-500">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">Upload documents to view insights</p>
          </div>
        </div>
      ) : (
        <>
          {/* Action Cards */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Quick Summary */}
            <Card className="p-4 ins-card hover:shadow-[0_0_15px_rgba(52,211,153,0.2)]">
              <div className="flex items-center gap-3 mb-3">
                <FileText className="h-5 w-5 accent-green" />
                <div>
                  <h3 className="font-semibold text-sm">Quick Summary</h3>
                  <p className="text-xs text-gray-400">Get a concise overview</p>
                </div>
              </div>
              <Button
                onClick={generateSummary}
                disabled={isGeneratingSummary}
                className="w-full btn-green"
                size="sm"
              >
                {isGeneratingSummary ? "Analyzing..." : "Generate Summary"}
              </Button>
            </Card>

            {/* Audio Overview */}
            <Card className="p-4 ins-card hover:shadow-[0_0_15px_rgba(129,140,248,0.2)]">
              <div className="flex items-center gap-3 mb-3">
                <Volume2 className="h-5 w-5 accent-purple" />
                <div>
                  <h3 className="font-semibold text-sm">Audio Overview</h3>
                  <p className="text-xs text-gray-400">Listen to the summary</p>
                </div>
              </div>
              <Button
                onClick={generateAudioOverview}
                disabled={isGeneratingAudio}
                className="w-full btn-purple"
                size="sm"
              >
                {isGeneratingAudio ? "Creating Audio..." : "Create Audio"}
              </Button>
              {!isGeneratingAudio && (
                <Button variant="ghost" size="sm" disabled className="w-full text-gray-500 mt-2">
                  <Play className="h-3 w-3 mr-2" /> No Audio Available
                </Button>
              )}
            </Card>

            {/* Risk Report */}
            <Card className="p-4 ins-card hover:shadow-[0_0_15px_rgba(248,113,113,0.2)]">
              <div className="flex items-center gap-3 mb-3">
                <BarChart3 className="h-5 w-5 accent-red" />
                <div>
                  <h3 className="font-semibold text-sm">Risk Report</h3>
                  <p className="text-xs text-gray-400">Detailed analysis with graphs</p>
                </div>
              </div>
              <Button
                onClick={generateDetailedReport}
                disabled={isGeneratingReport}
                className="w-full btn-red"
                size="sm"
              >
                {isGeneratingReport ? "Generating Report..." : "Generate Report"}
              </Button>
            </Card>

            {/* Personal Notes */}
            <Card className="p-4 ins-card hover:shadow-[0_0_15px_rgba(251,191,36,0.2)]">
              <div className="flex items-center gap-3 mb-3">
                <StickyNote className="h-5 w-5 accent-amber" />
                <div>
                  <h3 className="font-semibold text-sm">Personal Notes</h3>
                  <p className="text-xs text-gray-400">Add your thoughts</p>
                </div>
              </div>
              <Textarea
                value={userNotes}
                onChange={(e) => setUserNotes(e.target.value)}
                placeholder="Add your notes about this document..."
                className="min-h-[80px] text-xs resize-none bg-black/40 border border-gray-700 text-gray-200"
              />
              {userNotes.trim() && (
                <Button size="sm" className="w-full mt-2 btn-amber">
                  Save Notes
                </Button>
              )}
            </Card>

            {/* Recent Activity */}
            <Card className="p-4 ins-card">
              <h3 className="font-semibold text-sm mb-3">Recent Activity</h3>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">Summary</Badge>
                  <span className="text-gray-400">Generated 2 min ago</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">Analysis</Badge>
                  <span className="text-gray-400">Started 5 min ago</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-[rgba(255,255,255,0.08)]">
            <Button variant="outline" size="sm" className="w-full border-gray-600 text-gray-300 hover:bg-gray-800">
              <Download className="h-3 w-3 mr-2" /> Export All Insights
            </Button>
          </div>
        </>
      )}
    </div>
  );
};
