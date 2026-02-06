"""
Example usage script for the forecasting system
"""
import requests
import json
from pathlib import Path

# API base URL
API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✓ API is running")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure the API server is running.")
        print(f"  Start it with: python api.py")
        return False


def get_available_states():
    """Get list of available states"""
    try:
        response = requests.get(f"{API_BASE_URL}/states")
        if response.status_code == 200:
            data = response.json()
            print(f"\nAvailable states: {data['count']}")
            print(f"States: {', '.join(data['states'][:10])}{'...' if len(data['states']) > 10 else ''}")
            return data['states']
        else:
            print(f"Error getting states: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []


def get_forecast(state: str, weeks: int = 8):
    """Get forecast for a specific state"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/forecast",
            json={"state": state, "weeks": weeks}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Forecast generated for {state}")
            print(f"  Model used: {data['model_used']}")
            print(f"  Forecast horizon: {data['forecast_horizon_weeks']} weeks")
            print(f"  Number of predictions: {len(data['forecasts'])}")
            
            # Show first few predictions
            print(f"\n  First 5 predictions:")
            for i, forecast in enumerate(data['forecasts'][:5]):
                print(f"    {forecast['date']}: {forecast['predicted_sales']:.2f}")
            
            if len(data['forecasts']) > 5:
                print(f"    ... and {len(data['forecasts']) - 5} more")
            
            return data
        else:
            print(f"✗ Error getting forecast: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    print("="*70)
    print("Forecasting System - Example Usage")
    print("="*70)
    
    # Check API health
    if not check_api_health():
        return
    
    # Get available states
    states = get_available_states()
    
    if not states:
        print("\nNo states available. Please train models first using:")
        print("  python train.py")
        return
    
    # Get forecast for first state as example
    if states:
        example_state = states[0]
        print(f"\n{'='*70}")
        print(f"Example: Getting forecast for {example_state}")
        print(f"{'='*70}")
        forecast = get_forecast(example_state, weeks=8)
        
        # Save forecast to file
        if forecast:
            output_file = Path("example_forecast.json")
            with open(output_file, 'w') as f:
                json.dump(forecast, f, indent=2)
            print(f"\n✓ Forecast saved to: {output_file}")
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)
    print("\nTo get forecasts for other states, use:")
    print(f"  curl -X POST '{API_BASE_URL}/forecast' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"state\": \"StateName\", \"weeks\": 8}'")
    print("\nOr visit the API documentation at:")
    print(f"  {API_BASE_URL}/docs")


if __name__ == "__main__":
    main()

