import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, FileText, Shield, Brain, Zap, CheckCircle, Star } from "lucide-react";

const LandingPage = () => {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const featuresRef = useRef<HTMLElement>(null);
  const ctaRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
      
      // Scroll reveal animation
      const scrollRevealElements = document.querySelectorAll('.scroll-reveal');
      scrollRevealElements.forEach((element) => {
        const elementTop = element.getBoundingClientRect().top;
        const elementVisible = 150;
        
        if (elementTop < window.innerHeight - elementVisible) {
          element.classList.add('visible');
        }
      });
    };

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
    <div className="min-h-screen bg-background font-sans">
      {/* Hero Section with Enhanced Parallax */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 parallax-bg"
          style={{
            background: `linear-gradient(135deg, hsl(var(--primary) / 0.03) 0%, hsl(var(--accent) / 0.01) 100%)`,
            transform: `translateY(${scrollY * 0.3}px) scale(${1 + scrollY * 0.0002})`
          }}
        />
        
        {/* Animated background shapes */}
        <div 
          className="absolute inset-0 opacity-30"
          style={{
            transform: `translateY(${scrollY * 0.2}px) rotate(${scrollY * 0.05}deg)`
          }}
        >
          <div className="absolute top-20 left-1/4 w-64 h-64 bg-gradient-primary opacity-5 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-32 right-1/4 w-96 h-96 bg-gradient-primary opacity-3 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}} />
        </div>

        <div className="container mx-auto px-8 relative z-10">
          <div className="text-center max-w-5xl mx-auto">
            <div 
              className="opacity-0 animate-fade-in-up"
              style={{
                transform: `translateY(${scrollY * -0.1}px)`
              }}
            >
              <h1 className="heading-sans text-6xl md:text-7xl lg:text-8xl mb-8 font-light leading-tight">
                Simplify Complex
                <span className="bg-gradient-primary bg-clip-text text-transparent block mt-2 font-medium">
                  Legal Documents
                </span>
                <span className="text-5xl md:text-6xl lg:text-7xl block mt-4 font-light">
                  with AI
                </span>
              </h1>
            </div>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-16 max-w-3xl mx-auto leading-relaxed font-light opacity-0 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
              Upload your contracts, agreements, and legal documents. Get instant analysis, risk assessment, and plain-English explanations in seconds.
            </p>

            <div className="flex flex-col sm:flex-row gap-6 justify-center opacity-0 animate-fade-in-up" style={{animationDelay: '0.6s'}}>
              <Button 
                size="lg"
                className="bg-gradient-primary hover:shadow-glow-primary transition-smooth px-12 py-4 text-lg font-medium rounded-2xl"
                onClick={() => navigate("/chat")}
              >
                Try Legal Document Analysis
                <ArrowRight className="ml-3 h-5 w-5" />
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="border-2 border-border hover:bg-secondary transition-smooth px-12 py-4 text-lg font-medium rounded-2xl"
              >
                Watch Demo
              </Button>
            </div>

            <div className="flex items-center justify-center gap-8 mt-16 text-muted-foreground opacity-0 animate-fade-in-up" style={{animationDelay: '0.9s'}}>
              <div className="flex items-center gap-3">
                <CheckCircle className="h-5 w-5 text-primary" />
                <span className="text-base font-medium">Free Analysis</span>
              </div>
              <div className="flex items-center gap-3">
                <Shield className="h-5 w-5 text-primary" />
                <span className="text-base font-medium">Secure & Private</span>
              </div>
              <div className="flex items-center gap-3">
                <Star className="h-5 w-5 text-primary" />
                <span className="text-base font-medium">AI-Powered</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section ref={featuresRef} className="py-32 relative">
        <div 
          className="absolute inset-0 parallax-bg opacity-20"
          style={{
            transform: `translateY(${scrollY * 0.1}px)`
          }}
        >
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-border to-transparent" />
        </div>
        
        <div className="container mx-auto px-8 relative z-10">
          <div className="text-center mb-20 scroll-reveal">
            <h2 className="heading-sans text-5xl md:text-6xl mb-8 font-light">
              Powerful Legal Analysis
              <span className="bg-gradient-primary bg-clip-text text-transparent block mt-2 font-medium">
                Made Simple
              </span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto font-light leading-relaxed">
              Our AI analyzes your legal documents with the precision of a law firm and the speed of modern technology.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className="scroll-reveal group"
                style={{
                  transitionDelay: `${index * 150}ms`
                }}
              >
                <Card className="p-8 bg-card border-0 shadow-soft hover:shadow-elegant transition-smooth cursor-pointer h-full">
                  <div className="mb-6 p-4 bg-gradient-primary/5 rounded-2xl w-fit group-hover:bg-gradient-primary/10 transition-smooth">
                    <feature.icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="heading-sans text-xl mb-4 font-medium">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed font-light">{feature.description}</p>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section ref={ctaRef} className="py-32 relative overflow-hidden">
        <div 
          className="absolute inset-0 parallax-bg"
          style={{
            background: `linear-gradient(135deg, hsl(var(--primary) / 0.02) 0%, hsl(var(--accent) / 0.01) 100%)`,
            transform: `translateY(${scrollY * 0.15}px)`
          }}
        />
        
        <div className="container mx-auto px-8 text-center relative z-10">
          <div className="scroll-reveal">
            <h2 className="heading-sans text-5xl md:text-6xl mb-8 font-light">
              Ready to Understand Your
              <span className="bg-gradient-primary bg-clip-text text-transparent block mt-2 font-medium">
                Legal Documents?
              </span>
            </h2>
            <p className="text-xl text-muted-foreground mb-12 max-w-3xl mx-auto font-light leading-relaxed">
              Join thousands of professionals who trust our AI to analyze their legal documents quickly and accurately.
            </p>
            
            <Button 
              size="lg"
              className="bg-gradient-primary hover:shadow-glow-primary transition-smooth px-16 py-5 text-lg font-medium rounded-2xl animate-pulse"
              onClick={() => navigate("/chat")}
            >
              Start Analysis Now
              <ArrowRight className="ml-3 h-5 w-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-16 border-t border-border bg-gradient-hero/30">
        <div className="container mx-auto px-8 text-center">
          <p className="text-muted-foreground font-light text-lg">&copy; 2024 LegalAI. Revolutionizing legal document analysis with artificial intelligence.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;