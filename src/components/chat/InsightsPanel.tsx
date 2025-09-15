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
    // Simulate API call
    setTimeout(() => {
      setIsGeneratingSummary(false);
    }, 2000);
  };

  const generateAudioOverview = async () => {
    setIsGeneratingAudio(true);
    // Simulate API call
    setTimeout(() => {
      setIsGeneratingAudio(false);
    }, 3000);
  };

  const generateDetailedReport = async () => {
    setIsGeneratingReport(true);
    // Simulate API call
    setTimeout(() => {
      setIsGeneratingReport(false);
    }, 4000);
  };

  if (!hasDocuments) {
    return (
      <div className="w-80 bg-card border-l border-border flex flex-col h-full">
        <div className="p-6 border-b border-border">
          <h2 className="heading-serif text-xl">Insights</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="text-center text-muted-foreground">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">Upload documents to view insights</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-card border-l border-border flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <h2 className="heading-serif text-xl">Insights</h2>
        {activeDocument && (
          <p className="text-xs text-muted-foreground mt-1 truncate">
            {activeDocument}
          </p>
        )}
      </div>

      {/* Action Cards */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Quick Summary */}
        <Card className="p-4 hover:shadow-elegant transition-smooth">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <FileText className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-sm">Quick Summary</h3>
              <p className="text-xs text-muted-foreground">Get a concise overview</p>
            </div>
          </div>
          <Button 
            onClick={generateSummary}
            disabled={isGeneratingSummary}
            className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
            variant="outline"
            size="sm"
          >
            {isGeneratingSummary ? (
              <>
                <div className="w-3 h-3 border-2 border-primary/30 border-t-primary rounded-full animate-spin mr-2" />
                Analyzing...
              </>
            ) : (
              "Generate Summary"
            )}
          </Button>
        </Card>

        {/* Audio Overview */}
        <Card className="p-4 hover:shadow-elegant transition-smooth">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-accent/10 rounded-lg">
              <Volume2 className="h-4 w-4 text-accent" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-sm">Audio Overview</h3>
              <p className="text-xs text-muted-foreground">Listen to the summary</p>
            </div>
          </div>
          <div className="space-y-2">
            <Button 
              onClick={generateAudioOverview}
              disabled={isGeneratingAudio}
              className="w-full bg-accent/10 hover:bg-accent/20 text-accent border border-accent/20"
              variant="outline"
              size="sm"
            >
              {isGeneratingAudio ? (
                <>
                  <div className="w-3 h-3 border-2 border-accent/30 border-t-accent rounded-full animate-spin mr-2" />
                  Creating Audio...
                </>
              ) : (
                "Create Audio"
              )}
            </Button>
            {!isGeneratingAudio && (
              <Button 
                variant="ghost" 
                size="sm"
                className="w-full text-muted-foreground"
                disabled
              >
                <Play className="h-3 w-3 mr-2" />
                No Audio Available
              </Button>
            )}
          </div>
        </Card>

        {/* Detailed Report */}
        <Card className="p-4 hover:shadow-elegant transition-smooth">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-destructive/10 rounded-lg">
              <BarChart3 className="h-4 w-4 text-destructive" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-sm">Risk Report</h3>
              <p className="text-xs text-muted-foreground">Detailed analysis with graphs</p>
            </div>
          </div>
          <Button 
            onClick={generateDetailedReport}
            disabled={isGeneratingReport}
            className="w-full bg-destructive/10 hover:bg-destructive/20 text-destructive border border-destructive/20"
            variant="outline"
            size="sm"
          >
            {isGeneratingReport ? (
              <>
                <div className="w-3 h-3 border-2 border-destructive/30 border-t-destructive rounded-full animate-spin mr-2" />
                Generating Report...
              </>
            ) : (
              "Generate Report"
            )}
          </Button>
        </Card>

        {/* Add Notes */}
        <Card className="p-4 hover:shadow-elegant transition-smooth">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-muted rounded-lg">
              <StickyNote className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-sm">Personal Notes</h3>
              <p className="text-xs text-muted-foreground">Add your thoughts</p>
            </div>
          </div>
          <Textarea
            value={userNotes}
            onChange={(e) => setUserNotes(e.target.value)}
            placeholder="Add your notes about this document..."
            className="min-h-[80px] text-xs resize-none"
          />
          {userNotes.trim() && (
            <Button 
              size="sm"
              className="w-full mt-2 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
              variant="outline"
            >
              Save Notes
            </Button>
          )}
        </Card>

        {/* Recent Activity */}
        <Card className="p-4">
          <h3 className="font-semibold text-sm mb-3">Recent Activity</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <Badge variant="secondary" className="text-xs">Summary</Badge>
              <span className="text-muted-foreground">Generated 2 min ago</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <Badge variant="outline" className="text-xs">Analysis</Badge>
              <span className="text-muted-foreground">Started 5 min ago</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t border-border space-y-2">
        <Button 
          variant="outline" 
          size="sm"
          className="w-full gap-2"
        >
          <Download className="h-3 w-3" />
          Export All Insights
        </Button>
      </div>
    </div>
  );
};