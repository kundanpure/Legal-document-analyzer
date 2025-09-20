import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { FileText, Volume2, BarChart3, StickyNote, Download, Play } from "lucide-react";
import {
  useGenerateSummary,
  useGenerateAudio,
  useGenerateReport,
  useInsights,
  useExportConversation,
  useDownloadFile,
} from "@/hooks/api";

interface InsightsPanelProps {
  activeDocument: string | null;
  hasDocuments: boolean;
  activeFileId?: string | null;
  conversationId?: string | null;
}

type InsightWire = {
  summary?: { available?: boolean; url?: string; created_at?: string; word_count?: number };
  audio?: { available?: boolean; url?: string; created_at?: string; duration?: string };
  report?: { available?: boolean; url?: string; created_at?: string; page_count?: number };
  // Flat fallbacks some backends return:
  summary_available?: boolean;
  summary_url?: string;
  audio_available?: boolean;
  audio_url?: string;
  report_available?: boolean;
  report_url?: string;
  [k: string]: any;
};

export const InsightsPanel = ({
  activeDocument,
  hasDocuments,
  activeFileId,
  conversationId,
}: InsightsPanelProps) => {
  const [userNotes, setUserNotes] = useState("");

  // Actions
  const generateSummaryMutation = useGenerateSummary();
  const generateAudioMutation = useGenerateAudio();
  const generateReportMutation = useGenerateReport();
  const exportConversationMutation = useExportConversation();

  // Data
  const { data: insightsResp, refetch: refetchInsights } = useInsights(activeFileId ?? null);

  // Download helper (resolves backend base URL & triggers download)
  const downloadFile = useDownloadFile();

  // Normalize insights object (works for both nested and flat formats)
  const insights = (insightsResp?.insights || insightsResp || {}) as InsightWire;

  const summaryAvailable = useMemo(
    () => Boolean(insights.summary?.available ?? insights.summary_available),
    [insights]
  );
  const audioAvailable = useMemo(
    () => Boolean(insights.audio?.available ?? insights.audio_available),
    [insights]
  );
  const reportAvailable = useMemo(
    () => Boolean(insights.report?.available ?? insights.report_available),
    [insights]
  );

  const summaryUrl = insights.summary?.url ?? insights.summary_url ?? "";
  const audioUrl = insights.audio?.url ?? insights.audio_url ?? "";
  const reportUrl = insights.report?.url ?? insights.report_url ?? "";

  const generateSummary = async () => {
    if (!activeFileId) return;
    await generateSummaryMutation.mutateAsync({ fileId: activeFileId });
    setTimeout(() => refetchInsights(), 1000);
  };

  const generateAudioOverview = async () => {
    if (!activeFileId) return;
    await generateAudioMutation.mutateAsync({
      fileId: activeFileId,
      options: { voice_type: "female", language: "en", speed: 1.0 },
    });
    setTimeout(() => refetchInsights(), 1000);
  };

  const generateDetailedReport = async () => {
    if (!activeFileId) return;
    await generateReportMutation.mutateAsync({
      fileId: activeFileId,
      options: { type: "comprehensive", format: "pdf", language: "en" },
    });
    setTimeout(() => refetchInsights(), 1000);
  };

  const exportAllInsights = async () => {
    if (!conversationId) return;
    await exportConversationMutation.mutateAsync({ conversationId, format: "pdf" });
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
        .ins-card:hover { transform: translateY(-4px); }

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
        .btn-green:disabled {
          opacity: 0.5; cursor: not-allowed; background: rgba(52,211,153,0.06);
        }

        .btn-purple {
          background: rgba(129,140,248,0.12);
          border: 1px solid rgba(129,140,248,0.25);
          color: #818cf8;
        }
        .btn-purple:hover { background: rgba(129,140,248,0.18); }
        .btn-purple:disabled {
          opacity: 0.5; cursor: not-allowed; background: rgba(129,140,248,0.06);
        }

        .btn-red {
          background: rgba(248,113,113,0.12);
          border: 1px solid rgba(248,113,113,0.25);
          color: #f87171;
        }
        .btn-red:hover { background: rgba(248,113,113,0.18); }
        .btn-red:disabled {
          opacity: 0.5; cursor: not-allowed; background: rgba(248,113,113,0.06);
        }

        .btn-amber {
          background: rgba(251,191,36,0.12);
          border: 1px solid rgba(251,191,36,0.25);
          color: #fbbf24;
        }
        .btn-amber:hover { background: rgba(251,191,36,0.18); }

        .btn-success {
          background: rgba(34,197,94,0.12);
          border: 1px solid rgba(34,197,94,0.25);
          color: #22c55e;
        }
        .btn-success:hover { background: rgba(34,197,94,0.18); }
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

              {summaryAvailable ? (
                <div className="space-y-2">
                  <Badge variant="secondary" className="text-xs w-full justify-center">
                    Summary Available
                  </Badge>
                  <Button
                    onClick={() => {
                      if (!summaryUrl) return;
                      const filename = `summary_${activeFileId || "document"}.txt`; // backend serves text/plain
                      downloadFile(summaryUrl, filename);
                    }}
                    className="w-full btn-success"
                    size="sm"
                    disabled={!summaryUrl}
                  >
                    <Download className="h-3 w-3 mr-2" />
                    Download Summary
                  </Button>
                </div>
              ) : (
                <Button
                  onClick={generateSummary}
                  disabled={generateSummaryMutation.isPending || !activeFileId}
                  className="w-full btn-green"
                  size="sm"
                >
                  {generateSummaryMutation.isPending ? "Analyzing..." : "Generate Summary"}
                </Button>
              )}
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

              {audioAvailable ? (
                <div className="space-y-2">
                  <Badge variant="secondary" className="text-xs w-full justify-center">
                    Audio Available
                  </Badge>
                  <Button
                    onClick={() => {
                      if (!audioUrl) return;
                      const filename = `audio_${activeFileId || "document"}.mp3`;
                      downloadFile(audioUrl, filename);
                    }}
                    className="w-full btn-success"
                    size="sm"
                    disabled={!audioUrl}
                  >
                    <Play className="h-3 w-3 mr-2" />
                    Play Audio {insights.audio?.duration ? `(${insights.audio.duration})` : ""}
                  </Button>
                </div>
              ) : (
                <Button
                  onClick={generateAudioOverview}
                  disabled={generateAudioMutation.isPending || !activeFileId}
                  className="w-full btn-purple"
                  size="sm"
                >
                  {generateAudioMutation.isPending ? "Creating Audio..." : "Create Audio"}
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

              {reportAvailable ? (
                <div className="space-y-2">
                  <Badge variant="secondary" className="text-xs w-full justify-center">
                    Report Available {insights.report?.page_count ? `(${insights.report.page_count} pages)` : ""}
                  </Badge>
                  <Button
                    onClick={() => {
                      if (!reportUrl) return;
                      const filename = `report_${activeFileId || "document"}.pdf`;
                      downloadFile(reportUrl, filename);
                    }}
                    className="w-full btn-success"
                    size="sm"
                    disabled={!reportUrl}
                  >
                    <Download className="h-3 w-3 mr-2" />
                    Download Report
                  </Button>
                </div>
              ) : (
                <Button
                  onClick={generateDetailedReport}
                  disabled={generateReportMutation.isPending || !activeFileId}
                  className="w-full btn-red"
                  size="sm"
                >
                  {generateReportMutation.isPending ? "Generating Report..." : "Generate Report"}
                </Button>
              )}
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
                {summaryAvailable && (
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      Summary
                    </Badge>
                    <span className="text-gray-400">
                      Generated{" "}
                      {insights.summary?.created_at
                        ? new Date(insights.summary.created_at).toLocaleString()
                        : "recently"}
                    </span>
                  </div>
                )}
                {audioAvailable && (
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      Audio
                    </Badge>
                    <span className="text-gray-400">
                      Generated{" "}
                      {insights.audio?.created_at
                        ? new Date(insights.audio.created_at).toLocaleString()
                        : "recently"}
                    </span>
                  </div>
                )}
                {reportAvailable && (
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      Report
                    </Badge>
                    <span className="text-gray-400">
                      Generated{" "}
                      {insights.report?.created_at
                        ? new Date(insights.report.created_at).toLocaleString()
                        : "recently"}
                    </span>
                  </div>
                )}
                {!summaryAvailable && !audioAvailable && !reportAvailable && (
                  <p className="text-gray-500">No recent activity</p>
                )}
              </div>
            </Card>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-[rgba(255,255,255,0.08)]">
            <Button
              onClick={exportAllInsights}
              disabled={exportConversationMutation.isPending || !conversationId}
              variant="outline"
              size="sm"
              className="w-full border-gray-600 text-gray-300 hover:bg-gray-800"
            >
              <Download className="h-3 w-3 mr-2" />
              {exportConversationMutation.isPending ? "Exporting..." : "Export All Insights"}
            </Button>
            {!conversationId && (
              <p className="text-xs text-gray-500 mt-2 text-center">
                Start a conversation to enable export
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
};
