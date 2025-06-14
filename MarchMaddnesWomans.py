import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class MarchMadnessPredictor:
    def __init__(self):
        """
        Initialize the March Madness prediction model
        """
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_features(self, team_data):
        """
        Prepare features for machine learning model
        
        Expected columns:
        - Win percentage
        - Points scored per game
        - Points allowed per game
        - Rebounds per game
        - Assists per game
        - Turnover rate
        - Conference performance
        """
        features = [
            'win_percentage', 
            'points_scored', 
            'points_allowed', 
            'rebounds', 
            'assists', 
            'turnover_rate', 
            'conference_rank'
        ]
        
        # Ensure all required features exist
        for feature in features:
            if feature not in team_data.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        return team_data[features]
    
    def train_model(self, historical_data):
        """
        Train a machine learning model to predict tournament outcomes
        
        :param historical_data: DataFrame with team performance data and tournament results
        """
        # Prepare features and target
        X = self.prepare_features(historical_data)
        y = historical_data['tournament_performance']  # Binary: 1 for advanced, 0 for eliminated
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=5
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
    
    def predict_tournament_performance(self, team_data):
        """
        Predict a team's tournament performance
        
        :param team_data: DataFrame with a single team's data
        :return: Probability of advancing in the tournament
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare and scale features
        X = self.prepare_features(team_data)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities[:, 1]  # Probability of advancing
    
    def simulate_tournament(self, teams_data):
        """
        Simulate the entire tournament
        
        :param teams_data: DataFrame with all participating teams' data
        :return: Predicted tournament winner
        """
        # Predict performance for each team
        teams_data['advancement_prob'] = self.predict_tournament_performance(teams_data)
        
        # Sort teams by probability of advancement
        tournament_predictions = teams_data.sort_values('advancement_prob', ascending=False)
        
        print("\nTournament Predictions:")
        for index, row in tournament_predictions.iterrows():
            print(f"{row['team_name']}: {row['advancement_prob']*100:.2f}% chance of winning")
        
        return tournament_predictions.iloc[0]['team_name']

# Example usage
def main():
    # Create sample tournament data (this would be replaced with real historical data)
    np.random.seed(42)
    teams_data = pd.DataFrame({
        'team_name': [
            'UConn', 'LSU', 'South Carolina', 'Virginia Tech', 
            'Stanford', 'Notre Dame', 'Iowa', 'USC'
        ],
        'win_percentage': np.random.uniform(0.6, 0.9, 8),
        'points_scored': np.random.uniform(65, 85, 8),
        'points_allowed': np.random.uniform(50, 70, 8),
        'rebounds': np.random.uniform(35, 45, 8),
        'assists': np.random.uniform(15, 25, 8),
        'turnover_rate': np.random.uniform(0.1, 0.2, 8),
        'conference_rank': np.random.randint(1, 6, 8),
        'tournament_performance': np.random.randint(0, 2, 8)
    })
    
    # Initialize and train the model
    predictor = MarchMadnessPredictor()
    predictor.train_model(teams_data)
    
    # Simulate tournament
    winner = predictor.simulate_tournament(teams_data)
    print(f"\nPredicted Tournament Winner: {winner}")

if __name__ == "__main__":
    main()