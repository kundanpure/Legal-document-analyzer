import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, FileText, Shield, Brain, Zap, CheckCircle, Star } from "lucide-react";

const LandingPage = () => {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced AI understands complex legal language and identifies key clauses, risks, and opportunities in seconds."
    },
    {
      icon: Shield,
      title: "Risk Assessment",
      description: "Automatically detect potential legal risks, hidden clauses, and unfavorable terms before you sign."
    },
    {
      icon: Zap,
      title: "Instant Insights",
      description: "Get immediate summaries, explanations, and actionable insights from your legal documents."
    },
    {
      icon: FileText,
      title: "Multi-Document Support",
      description: "Analyze contracts, agreements, terms & conditions, and more. Support for up to 50 documents simultaneously."
    }
  ];

  return (
    <div className="min-h-screen bg-background font-body">
      {/* Hero Section with Parallax */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 bg-gradient-hero opacity-50"
          style={{
            transform: `translateY(${scrollY * 0.5}px)`
          }}
        />
        <div className="container mx-auto px-6 relative z-10">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="heading-serif text-5xl md:text-7xl lg:text-8xl mb-8 animate-fade-in-up">
              Simplify Complex
              <span className="bg-gradient-primary bg-clip-text text-transparent block mt-4">
                Legal Documents
              </span>
              <span className="text-4xl md:text-6xl lg:text-7xl block mt-4">
                with AI
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              Upload your contracts, agreements, and legal documents. Get instant analysis, risk assessment, and plain-English explanations.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <Button 
                size="lg"
                className="bg-gradient-primary hover:shadow-glow-primary transition-smooth px-8 py-6 text-lg font-semibold"
                onClick={() => navigate("/chat")}
              >
                Try Legal Document Analysis
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="border-border hover:bg-secondary transition-smooth px-8 py-6 text-lg"
              >
                Watch Demo
              </Button>
            </div>

            <div className="flex items-center justify-center gap-6 mt-12 text-muted-foreground animate-fade-in-up" style={{animationDelay: '0.6s'}}>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                <span className="text-sm">Free Analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                <span className="text-sm">Secure & Private</span>
              </div>
              <div className="flex items-center gap-2">
                <Star className="h-4 w-4 text-primary" />
                <span className="text-sm">AI-Powered</span>
              </div>
            </div>
          </div>
        </div>

        {/* Floating Elements */}
        <div className="absolute top-1/4 left-10 w-20 h-20 bg-primary/10 rounded-full animate-float" style={{animationDelay: '0s'}} />
        <div className="absolute top-1/3 right-16 w-12 h-12 bg-accent/10 rounded-full animate-float" style={{animationDelay: '2s'}} />
        <div className="absolute bottom-1/4 left-1/4 w-16 h-16 bg-primary/5 rounded-full animate-float" style={{animationDelay: '4s'}} />
      </section>

      {/* Features Section */}
      <section className="py-24 bg-card/30">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="heading-serif text-4xl md:text-5xl mb-6">
              Powerful Legal Analysis
              <span className="bg-gradient-primary bg-clip-text text-transparent block">
                Made Simple
              </span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Our AI analyzes your legal documents with the precision of a law firm and the speed of modern technology.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Card 
                key={feature.title}
                className="p-6 bg-card border-border hover:shadow-elegant transition-smooth animate-fade-in-up cursor-pointer group"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className="mb-4 p-3 bg-primary/10 rounded-lg w-fit group-hover:bg-primary/20 transition-smooth">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="heading-serif text-xl mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-hero">
        <div className="container mx-auto px-6 text-center">
          <h2 className="heading-serif text-4xl md:text-5xl mb-6">
            Ready to Understand Your
            <span className="bg-gradient-primary bg-clip-text text-transparent block mt-2">
              Legal Documents?
            </span>
          </h2>
          <p className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto">
            Join thousands of professionals who trust our AI to analyze their legal documents quickly and accurately.
          </p>
          
          <Button 
            size="lg"
            className="bg-gradient-primary hover:shadow-glow-primary transition-smooth px-10 py-6 text-lg font-semibold animate-glow"
            onClick={() => navigate("/chat")}
          >
            Start Analysis Now
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="container mx-auto px-6 text-center text-muted-foreground">
          <p>&copy; 2024 LegalAI. Revolutionizing legal document analysis with artificial intelligence.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;