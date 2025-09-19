import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Home, ArrowLeft, Search } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-gray-900 to-black">
      <style>{`
        .glow-text {
          background: linear-gradient(90deg, #f8fafc, #ffd37b);
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          text-shadow: 0 0 20px rgba(255, 211, 123, 0.3);
        }
        .glass-card {
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 20px;
          backdrop-filter: blur(10px);
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        .floating-orb {
          position: absolute;
          width: 200px;
          height: 200px;
          border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, rgba(107, 124, 255, 0.1), transparent);
          filter: blur(40px);
          animation: float 6s ease-in-out infinite;
        }
        .floating-orb:nth-child(1) {
          top: 10%;
          left: 20%;
          animation-delay: -2s;
        }
        .floating-orb:nth-child(2) {
          top: 60%;
          right: 20%;
          animation-delay: -4s;
          background: radial-gradient(circle at 30% 30%, rgba(155, 214, 255, 0.08), transparent);
        }
        .floating-orb:nth-child(3) {
          bottom: 20%;
          left: 30%;
          animation-delay: -1s;
          background: radial-gradient(circle at 30% 30%, rgba(255, 211, 123, 0.06), transparent);
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }
      `}</style>

      {/* Floating Background Orbs */}
      <div className="floating-orb"></div>
      <div className="floating-orb"></div>
      <div className="floating-orb"></div>

      <div className="text-center max-w-lg mx-auto px-6 relative z-10">
        <div className="glass-card p-12">
          {/* 404 Number */}
          <div className="mb-8">
            <h1 className="text-8xl md:text-9xl font-bold glow-text mb-4">
              404
            </h1>
            <div className="w-24 h-1 bg-gradient-to-r from-transparent via-yellow-400 to-transparent mx-auto opacity-60"></div>
          </div>

          {/* Error Message */}
          <h2 className="text-2xl md:text-3xl font-semibold text-white mb-4">
            Page Not Found
          </h2>
          
          <p className="text-gray-400 mb-2 text-lg">
            The page you're looking for doesn't exist.
          </p>
          
          <p className="text-sm text-gray-500 mb-8">
            Path: <span className="font-mono bg-gray-800 px-2 py-1 rounded text-gray-300">{location.pathname}</span>
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/landing">
              <Button 
                className="w-full sm:w-auto bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white border-0 px-8 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
              >
                <Home className="w-4 h-4 mr-2" />
                Go Home
              </Button>
            </Link>
            
            <Button 
              variant="outline" 
              className="w-full sm:w-auto border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white px-8 py-3 rounded-lg"
              onClick={() => window.history.back()}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Go Back
            </Button>
          </div>

          {/* Additional Help */}
          <div className="mt-8 pt-6 border-t border-gray-800">
            <p className="text-sm text-gray-500 mb-4">Need help? Try these:</p>
            <div className="flex flex-wrap justify-center gap-2">
              <Link 
                to="/chat" 
                className="text-blue-400 hover:text-blue-300 text-sm underline transition-colors"
              >
                Chat with AI
              </Link>
              <span className="text-gray-600">•</span>
              <Link 
                to="/landing" 
                className="text-blue-400 hover:text-blue-300 text-sm underline transition-colors"
              >
                Learn More
              </Link>
              <span className="text-gray-600">•</span>
              <a 
                href="mailto:support@legalai.com" 
                className="text-blue-400 hover:text-blue-300 text-sm underline transition-colors"
              >
                Contact Support
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;