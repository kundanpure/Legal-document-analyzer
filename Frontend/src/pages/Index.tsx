import { Navigate } from "react-router-dom";

const Index = () => {
  // Redirect to landing page instead of chat for better UX
  return <Navigate to="/landing" replace />;
};

export default Index;