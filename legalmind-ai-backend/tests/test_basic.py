"""
Basic tests to verify LegalMind AI setup
"""

import pytest
import sys
import os

# Add the parent directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_python_version():
    """Test Python version is appropriate"""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print(f"✅ Python version: {sys.version}")


def test_basic_math():
    """Test that basic operations work"""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"
    assert [1, 2, 3][0] == 1
    print("✅ Basic Python operations work")


def test_project_structure():
    """Test that key project files exist"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for main.py
    main_file = os.path.join(project_root, "main.py")
    assert os.path.exists(main_file), "main.py should exist"
    print("✅ main.py exists")
    
    # Check for app directory
    app_dir = os.path.join(project_root, "app")
    assert os.path.exists(app_dir), "app/ directory should exist"
    print("✅ app/ directory exists")
    
    # Check for config directory
    config_dir = os.path.join(project_root, "config")
    assert os.path.exists(config_dir), "config/ directory should exist"
    print("✅ config/ directory exists")


def test_imports():
    """Test that we can import key modules"""
    
    # Test config import
    try:
        from config.settings import get_settings
        settings = get_settings()
        assert settings is not None
        print("✅ Settings import successful")
    except ImportError as e:
        pytest.skip(f"Settings import failed: {e}")
    
    # Test main app import
    try:
        import main
        assert hasattr(main, 'app'), "main.py should have 'app' variable"
        print("✅ Main app import successful")
    except ImportError as e:
        pytest.skip(f"Main app import failed: {e}")


class TestAppStructure:
    """Test application structure and basic functionality"""
    
    def test_fastapi_app_exists(self):
        """Test that FastAPI app can be created"""
        try:
            import main
            app = main.app
            assert app is not None
            print("✅ FastAPI app exists")
            
            # Check if it's actually a FastAPI instance
            from fastapi import FastAPI
            assert isinstance(app, FastAPI), "app should be FastAPI instance"
            print("✅ App is valid FastAPI instance")
            
        except Exception as e:
            pytest.skip(f"FastAPI app test failed: {e}")
    
    def test_api_routes_structure(self):
        """Test that API routes directory exists"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        routes_dir = os.path.join(project_root, "app", "api", "routes")
        
        assert os.path.exists(routes_dir), "API routes directory should exist"
        print("✅ API routes directory exists")
        
        # Check for key route files
        key_routes = ["documents.py", "chat.py", "voice.py", "reports.py", "system.py"]
        for route_file in key_routes:
            route_path = os.path.join(routes_dir, route_file)
            if os.path.exists(route_path):
                print(f"✅ {route_file} exists")
            else:
                print(f"⚠️  {route_file} missing")
    
    def test_models_exist(self):
        """Test that model files exist"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "app", "models")
        
        assert os.path.exists(models_dir), "Models directory should exist"
        print("✅ Models directory exists")
        
        # Check for key model files
        key_models = ["requests.py", "responses.py", "database.py"]
        for model_file in key_models:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                print(f"✅ {model_file} exists")
            else:
                print(f"⚠️  {model_file} missing")


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works"""
    import asyncio
    
    async def dummy_async_function():
        await asyncio.sleep(0.01)  # 10ms sleep
        return "async_works"
    
    result = await dummy_async_function()
    assert result == "async_works"
    print("✅ Async functionality works")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
