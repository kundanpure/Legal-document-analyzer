import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, Upload, Brain, Shield, Star, Lock, Eye, Database, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";

const VideoPlayer = ({ src }: { src: string }) => {
  const ref = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!el) return;
          if (entry.isIntersecting) el.play().catch(() => {});
          else el.pause();
        });
      },
      { threshold: 0.55 }
    );
    obs.observe(el);
    return () => {
      obs.disconnect();
      el.pause();
    };
  }, []);

  return (
    <video
      ref={ref}
      src={src}
      className="w-full h-64 md:h-80 rounded-2xl object-cover"
      muted
      playsInline
      preload="metadata"
      loop
      controls={false}
      aria-hidden
    />
  );
};

const FloatingElement = ({ delay = 0, duration = 4, size = 80, blur = 20, opacity = 0.05, color = "#6b7cff" }) => {
  return (
    <div 
      className="absolute rounded-full pointer-events-none"
      style={{
        width: size,
        height: size,
        background: color,
        filter: `blur(${blur}px)`,
        opacity: opacity,
        animation: `float-${duration}s ${delay}s ease-in-out infinite alternate`
      }}
    />
  );
};

const LandingPage = () => {
  const [scrollY] = useState(0);
  const testimonialsRef = useRef<HTMLDivElement | null>(null);
  const testimonialsObserverRef = useRef<IntersectionObserver | null>(null);
  const rafRef = useRef<number | null>(null);
  const scrollPosRef = useRef(0);
  const navigate = useNavigate();


  useEffect(() => {
    const els = document.querySelectorAll(".scroll-reveal");
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry, i) => {
          if (entry.isIntersecting) (entry.target as HTMLElement).classList.add("visible");
        });
      },
      { threshold: 0.12 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const el = testimonialsRef.current;
    if (!el) return;
    el.scrollLeft = 0;
    const start = () => {
      const step = () => {
        scrollPosRef.current += 0.5;
        if (scrollPosRef.current >= el.scrollWidth / 2) scrollPosRef.current = 0;
        el.scrollLeft = scrollPosRef.current;
        rafRef.current = requestAnimationFrame(step);
      };
      rafRef.current = requestAnimationFrame(step);
    };
    const stop = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
    testimonialsObserverRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) start();
          else stop();
        });
      },
      { threshold: 0.6 }
    );
    testimonialsObserverRef.current.observe(el);
    return () => {
      stop();
      testimonialsObserverRef.current?.disconnect();
    };
  }, []);

  const features = [
    {
      icon: Upload,
      title: "Upload Sources",
      description: "Drag and drop your legal PDFs instantly",
      isVideo: true,
      videoSrc: "https://notebooklm.google/_/static/v4/videos/upload_your_sources.mp4",
    },
    {
      icon: Brain,
      title: "Instant Insights",
      description: "AI analyzes and summarizes in seconds",
      isVideo: true,
      videoSrc: "https://notebooklm.google/_/static/v4/videos/see_the_source_not_just_the_answer.mp4",
    },
    {
      icon: Shield,
      title: "Secure & Private",
      description: "Your documents stay completely private",
      isVideo: true,
      videoSrc: "https://notebooklm.google/_/static/v4/videos/listen_and_learn_on_the_go.mp4",
    },
  ];

  const testimonials = [
    { name: "Ayesha R.", role: "Corporate Counsel", quote: "LegalAI turned a 40-page contract into clear risks in seconds.", avatar: "AR", rating: 5 },
    { name: "Mark T.", role: "Startup Founder", quote: "Saved us hours of legal review; incredibly accurate summaries.", avatar: "MT", rating: 5 },
    { name: "Sofia L.", role: "Senior Paralegal", quote: "The insights are concise and trustworthy — a daily tool now.", avatar: "SL", rating: 5 },
    { name: "Daniel K.", role: "Chief Operations Officer", quote: "Privacy-first approach gave our team confidence to upload docs.", avatar: "DK", rating: 5 },
    { name: "Rachel M.", role: "Legal Director", quote: "Transformed our document review process completely.", avatar: "RM", rating: 5 },
    { name: "James P.", role: "General Counsel", quote: "Best AI tool for legal work I've used. Highly recommend.", avatar: "JP", rating: 5 },
  ];

  const privacyFeatures = [
    { icon: Lock, title: "End-to-End Encryption", desc: "Military-grade encryption protects your data" },
    { icon: Eye, title: "Zero Data Training", desc: "We never use your documents to train AI models" },
    { icon: Database, title: "No Data Retention", desc: "Files deleted after your session ends" },
  ];

  return (
    <div className="min-h-screen bg-black font-sans relative overflow-x-hidden text-white">
      <style>{`
        :root { --muted: #9aa3ad; --glass-border: rgba(255,255,255,0.06); --fg: #eef2f6; }
        * { box-sizing: border-box; }
        body,html,#root { background: #000; }
        nav { position:fixed; top:0; width:100%; z-index:60; backdrop-filter: blur(8px); background: rgba(0,0,0,0.6); border-bottom: 1px solid rgba(255,255,255,0.04); }
        .container { max-width:1200px; margin:0 auto; padding:0 1.5rem; }
        .hero { padding-top:92px; padding-bottom:36px; display:flex; align-items:center; justify-content:center; min-height:520px; text-align:center; }
        .headline { margin:0; line-height:1.02; font-weight:300; }
        .line-1 { font-size:3.5rem; color:var(--fg); display:block; }
        .line-2 { font-size:4.75rem; display:block; font-weight:700; background: linear-gradient(90deg,#f8fafc,#ffd37b); -webkit-background-clip:text; background-clip:text; color:transparent; margin-top:6px; letter-spacing:-1px; }
        .line-3 { font-size:3.5rem; display:block; color:var(--fg); margin-top:6px; font-weight:700; position:relative; display:inline-block; }
        .ai-underline, .privacy-underline { position:absolute; left:50%; transform:translateX(-50%); bottom:-8px; width:96px; height:22px; pointer-events:none; }
        .sub { color:var(--muted); margin-top:18px; max-width:820px; margin-left:auto; margin-right:auto; font-size:1.05rem; }
        .liquid-btn { position:relative; padding:16px 32px; border-radius:28px; border:1px solid var(--glass-border); background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color:var(--fg); overflow:hidden; backdrop-filter: blur(8px); display:inline-flex; gap:10px; align-items:center; transition: transform .18s ease, box-shadow .18s ease; }
        .liquid-btn:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.6); }
        .liquid-shine { position:absolute; top:-50%; left:-30%; width:140%; height:220%; background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.06), transparent 12%), linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0)); transform: rotate(-18deg); transition: transform .6s ease; pointer-events:none; mix-blend-mode: overlay; }
        .sections { padding-bottom:12px; }
        .feature { padding:42px 0; display:flex; align-items:center; gap:36px; }
        .feature .left { max-width:560px; }
        .feature .icon-wrap { width:84px; height:84px; border-radius:18px; display:grid; place-items:center; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); margin-bottom:18px; }
        .card-preview { border-radius:16px; border:1px solid rgba(255,255,255,0.06); overflow:hidden; background: #06060a; }
        .compact { padding:18px; }
        
        /* Modern Privacy Section */
        .privacy-section { 
          padding: 80px 0; 
          position: relative; 
          background: radial-gradient(ellipse at center, rgba(107, 124, 255, 0.03) 0%, transparent 70%);
          overflow: hidden;
        }
        .privacy-grid { 
          display: grid; 
          grid-template-columns: 1fr 1fr; 
          gap: 60px; 
          align-items: center; 
          position: relative;
          z-index: 2;
        }
        .privacy-content {
          max-width: 480px;
        }
        .privacy-visual {
          position: relative;
          height: 400px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .privacy-center {
          width: 120px;
          height: 120px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
          position: relative;
          z-index: 10;
        }
        .privacy-ring {
          position: absolute;
          border: 2px solid rgba(255, 255, 255, 0.1);
          border-radius: 50%;
          animation: spin 20s linear infinite;
        }
        .privacy-ring:nth-child(1) { width: 200px; height: 200px; animation-duration: 25s; }
        .privacy-ring:nth-child(2) { width: 280px; height: 280px; animation-duration: 35s; animation-direction: reverse; }
        .privacy-ring:nth-child(3) { width: 360px; height: 360px; animation-duration: 45s; }
        
        .privacy-features {
          display: grid;
          gap: 24px;
          margin-top: 32px;
        }
        .privacy-feature {
          display: flex;
          align-items: flex-start;
          gap: 16px;
          padding: 20px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          transition: all 0.3s ease;
        }
        .privacy-feature:hover {
          background: rgba(255, 255, 255, 0.04);
          transform: translateY(-2px);
        }
        .privacy-icon {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          background: linear-gradient(135deg, rgba(107, 124, 255, 0.2), rgba(155, 214, 255, 0.2));
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        /* Modern Testimonials */
        .testimonials-section {
          padding: 80px 0;
          background: linear-gradient(180deg, transparent 0%, rgba(107, 124, 255, 0.02) 50%, transparent 100%);
          position: relative;
          overflow: hidden;
        }
        .testimonials-header {
          text-align: center;
          margin-bottom: 60px;
        }
        .testimonials-track {
          display: flex;
          gap: 24px;
          padding: 20px;
          overflow-x: hidden;
          position: relative;
        }
        .testimonial-card {
          flex: 0 0 380px;
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          padding: 32px;
          backdrop-filter: blur(10px);
          position: relative;
          overflow: hidden;
        }
        .testimonial-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        }
        .testimonial-header {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 20px;
        }
        .testimonial-avatar {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 600;
          font-size: 16px;
        }
        .testimonial-info h4 {
          margin: 0;
          font-size: 16px;
          color: #eef2f6;
          font-weight: 600;
        }
        .testimonial-info p {
          margin: 4px 0 0 0;
          font-size: 14px;
          color: var(--muted);
        }
        .testimonial-stars {
          display: flex;
          gap: 2px;
          margin-bottom: 16px;
        }
        .testimonial-quote {
          color: #eef2f6;
          font-size: 16px;
          line-height: 1.6;
          position: relative;
        }
        .testimonial-quote::before {
          content: '"';
          position: absolute;
          top: -10px;
          left: -8px;
          font-size: 40px;
          color: rgba(107, 124, 255, 0.3);
          font-family: serif;
        }

        .scroll-reveal { opacity:0; transform: translateY(18px); transition: all 600ms cubic-bezier(.2,.9,.3,1); }
        .scroll-reveal.visible { opacity:1; transform: translateY(0); }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes float-4s { 0% { transform: translateY(0px) rotate(0deg); } 100% { transform: translateY(-20px) rotate(180deg); } }
        @keyframes float-6s { 0% { transform: translateY(0px) rotate(0deg); } 100% { transform: translateY(-30px) rotate(-90deg); } }
        @keyframes float-8s { 0% { transform: translateY(0px) rotate(0deg); } 100% { transform: translateY(-15px) rotate(270deg); } }

        @media (max-width:900px){ 
          .line-2{ font-size:2.6rem } 
          .line-1,.line-3{ font-size:1.8rem } 
          .feature{ flex-direction:column-reverse; padding:28px 0 }
          .privacy-grid { grid-template-columns: 1fr; gap: 40px; text-align: center; }
          .privacy-visual { height: 300px; }
          .privacy-section { padding: 60px 0; }
          .testimonials-section { padding: 60px 0; }
          .testimonial-card{ flex:0 0 300px }
          .container { padding: 0 1rem; }
          .hero { padding-top: 80px; padding-bottom: 24px; min-height: 400px; }
        }
        
        @media (max-width:768px){
          .line-1, .line-3 { font-size: 1.6rem; }
          .line-2 { font-size: 2.2rem; }
          .privacy-content h2 { font-size: 28px; }
          .testimonials-header h2 { font-size: 28px; }
          .privacy-features { gap: 16px; }
          .privacy-feature { padding: 16px; }
          .testimonial-card { flex: 0 0 280px; padding: 24px; }
          .feature .left { text-align: center; }
          .feature .container { grid-template-columns: 1fr !important; }
        }
        
        @media (max-width:480px){
          .line-1, .line-3 { font-size: 1.4rem; }
          .line-2 { font-size: 1.8rem; }
          .privacy-content h2 { font-size: 24px; }
          .testimonials-header h2 { font-size: 24px; }
          .privacy-section, .testimonials-section { padding: 40px 0; }
          .testimonial-card { flex: 0 0 260px; padding: 20px; }
          .privacy-visual { height: 250px; }
          .privacy-ring:nth-child(1) { width: 150px; height: 150px; }
          .privacy-ring:nth-child(2) { width: 200px; height: 200px; }
          .privacy-ring:nth-child(3) { width: 250px; height: 250px; }
          .privacy-center { width: 80px; height: 80px; }
          .nav { padding: 0.5rem 0; }
        }
      `}</style>

      <nav>
        <div className="container" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem 0" }}>
          <div style={{ fontSize: 18, fontWeight: 600 }}>LegalAI</div>
          <div style={{ display: "flex", gap: 24, alignItems: "center", color: "var(--muted)" }} className="hidden md:flex">
            <span style={{ cursor: "pointer" }}>Overview</span>
            <span style={{ cursor: "pointer" }}>Features</span>
            <span style={{ cursor: "pointer" }}>Contact</span>
          </div>
        </div>
      </nav>

      <header className="hero">
        <div className="container">
          <h1 className="headline" aria-hidden>
            <span className="line-1">Simplify Complex</span>
            <span className="line-2">Legal Documents</span>
            <span className="line-3">With <span style={{ position: "relative", display: "inline-block" }}>AI
              <svg className="ai-underline" viewBox="0 0 200 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                <path d="M5 28 C40 5, 120 35, 195 20" fill="none" stroke="#ffd37b" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" opacity="0.95"/>
                <path d="M5 30 C40 10, 120 37, 195 22" fill="none" stroke="#f8fafc" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.18"/>
              </svg>
            </span></span>
          </h1>

          <p className="sub">AI that helps you analyze, summarize, and reason with your legal PDFs — fast, private, and reliable.</p>

          <div style={{ marginTop: 24 }}>
            <button className="liquid-btn" onClick={() => navigate("/chat")} aria-label="Try Now">
              <span style={{ position: "relative", zIndex: 10, display: "inline-flex", alignItems: "center", gap: 8 }}>Try Now <ArrowRight /></span>
              <span className="liquid-shine" />
            </button>
          </div>
        </div>
      </header>

      <main className="sections">
        {features.map((feature, i) => (
          <section key={feature.title} className="feature scroll-reveal" style={{ background: i === 0 ? "#000" : "transparent" }}>
            <div className="container" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 36, alignItems: "center" }}>
              <div className="left">
                <div className="icon-wrap"><feature.icon style={{ width: 36, height: 36, color: "#fff" }} /></div>
                <h3 style={{ fontSize: 32, margin: 0, color: "#eef2f6", fontWeight: 300 }}>{feature.title}</h3>
                <p style={{ color: "var(--muted)", marginTop: 12 }}>{feature.description}</p>
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 14, color: "var(--muted)" }}>
                  <Star style={{ width: 16, height: 16, color: "#cde7ff" }} />
                  <span>Powered by advanced AI technology</span>
                </div>
              </div>

              <div>
                <Card className="card-preview" style={{ padding: 0 }}>
                  <div className="compact">
                    {feature.isVideo ? <VideoPlayer src={feature.videoSrc!} /> : <div className="h-64 md:h-80 flex items-center justify-content-center text-gray-500">{feature.title} Preview</div>}
                  </div>
                </Card>
              </div>
            </div>
          </section>
        ))}

        {/* Modern Privacy Section */}
        <section className="privacy-section scroll-reveal">
          {/* Floating Background Elements */}
          <div style={{ position: "absolute", top: "10%", left: "5%", animation: "float-6s 0s ease-in-out infinite alternate" }}>
            <FloatingElement size={60} blur={25} opacity={0.06} color="#6b7cff" />
          </div>
          <div style={{ position: "absolute", top: "60%", right: "8%", animation: "float-4s 1s ease-in-out infinite alternate" }}>
            <FloatingElement size={80} blur={20} opacity={0.08} color="#9bd6ff" />
          </div>
          <div style={{ position: "absolute", top: "20%", right: "20%", animation: "float-8s 2s ease-in-out infinite alternate" }}>
            <FloatingElement size={40} blur={15} opacity={0.04} color="#667eea" />
          </div>
          <div style={{ position: "absolute", bottom: "30%", left: "15%", animation: "float-6s 1.5s ease-in-out infinite alternate" }}>
            <FloatingElement size={100} blur={30} opacity={0.05} color="#764ba2" />
          </div>

          <div className="container">
            <div className="privacy-grid">
              <div className="privacy-content">
                <h2 style={{ fontSize: 42, fontWeight: 700, color: "#eef2f6", margin: 0, lineHeight: 1.2 }}>
                  Your Privacy is <span style={{ position: "relative", display: "inline-block", background: "linear-gradient(90deg,#f8fafc,#ffd37b)", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Sacred
                    <svg style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", bottom: "-8px", width: "120px", height: "22px", pointerEvents: "none" }} viewBox="0 0 200 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                      <path d="M5 28 C40 5, 120 35, 195 20" fill="none" stroke="#ffd37b" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" opacity="0.95"/>
                      <path d="M5 30 C40 10, 120 37, 195 22" fill="none" stroke="#f8fafc" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.18"/>
                    </svg>
                  </span>
                </h2>
                <p style={{ color: "var(--muted)", marginTop: 16, fontSize: 18, lineHeight: 1.6 }}>
                  Built with privacy-first principles. Your sensitive legal documents are protected by enterprise-grade security at every step.
                </p>
                
                <div className="privacy-features">
                  {privacyFeatures.map((feature, index) => (
                    <div key={index} className="privacy-feature">
                      <div className="privacy-icon">
                        <feature.icon style={{ width: 20, height: 20, color: "#667eea" }} />
                      </div>
                      <div>
                        <h4 style={{ margin: 0, fontSize: 16, color: "#eef2f6", fontWeight: 600 }}>{feature.title}</h4>
                        <p style={{ margin: "4px 0 0 0", fontSize: 14, color: "var(--muted)" }}>{feature.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="privacy-visual">
                <div className="privacy-ring"></div>
                <div className="privacy-ring"></div>
                <div className="privacy-ring"></div>
                <div className="privacy-center">
                  <Shield style={{ width: 40, height: 40, color: "white" }} />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Modern Testimonials Section */}
        <section className="testimonials-section scroll-reveal">
          <div className="container">
            <div className="testimonials-header">
              <h2 style={{ fontSize: 42, fontWeight: 700, color: "#eef2f6", margin: 0 }}>
                Trusted by <span style={{ position: "relative", display: "inline-block", background: "linear-gradient(90deg,#f8fafc,#ffd37b)", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Legal Professionals
                  <svg style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", bottom: "-8px", width: "280px", height: "22px", pointerEvents: "none" }} viewBox="0 0 340 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                    <path d="M5 28 C80 5, 260 35, 335 20" fill="none" stroke="#ffd37b" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" opacity="0.95"/>
                    <path d="M5 30 C80 10, 260 37, 335 22" fill="none" stroke="#f8fafc" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.18"/>
                  </svg>
                </span>
              </h2>
              <p style={{ color: "var(--muted)", marginTop: 16, fontSize: 18 }}>
                Join thousands of lawyers, paralegals, and legal teams who trust LegalAI
              </p>
            </div>
            
            <div ref={testimonialsRef} className="testimonials-track">
              {Array.from({ length: 2 }).flatMap(() => testimonials).map((testimonial, idx) => (
                <div key={idx} className="testimonial-card">
                  <div className="testimonial-header">
                    <div className="testimonial-avatar">
                      {testimonial.avatar}
                    </div>
                    <div className="testimonial-info">
                      <h4>{testimonial.name}</h4>
                      <p>{testimonial.role}</p>
                    </div>
                  </div>
                  
                  <div className="testimonial-stars">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} style={{ width: 16, height: 16, color: "#ffd37b", fill: "#ffd37b" }} />
                    ))}
                  </div>
                  
                  <div className="testimonial-quote">
                    {testimonial.quote}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer style={{ padding: "36px 0", borderTop: "1px solid rgba(255,255,255,0.04)", textAlign: "center", color: "var(--muted)" }}>
        &copy; {new Date().getFullYear()} LegalAI. Transforming legal document analysis with AI.
      </footer>
    </div>
  );
};

export default LandingPage;