# End-to-End Market Basket Analysis Demonstration
# Complete workflow showcasing all implemented components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import time
import json
from dataclasses import dataclass

# Configuration for demonstration
DEMO_CONFIG = {
    'use_prototype_data': True,
    'sample_size': 5000,
    'rating_threshold': 4.0,
    'min_support': 0.02,
    'min_confidence': 0.6,
    'num_recommendations': 10,
    'demo_users': ['demo_user_1', 'demo_user_2', 'demo_user_3']
}

@dataclass
class DemoResults:
    """Container for demonstration results and insights."""
    dataset_stats: Dict[str, Any]
    preprocessing_stats: Dict[str, Any]
    mining_results: Dict[str, Any]
    recommendation_results: Dict[str, Any]
    business_insights: List[str]
    performance_metrics: Dict[str, Any]

class MarketBasketAnalysisDemo:
    """
    Complete end-to-end demonstration of the market basket analysis system.
    Showcases data loading, mining, recommendation generation, and business insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEMO_CONFIG
        self.results = None
        self.demo_data = None
        
    def generate_demo_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate realistic demo data for the demonstration."""
        print("📊 Generating realistic demo dataset...")
        
        # Generate book metadata
        genres = ['Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 'Biography', 
                 'History', 'Self-Help', 'Business', 'Technology']
        
        books_data = []
        for i in range(200):
            book_id = f"book_{i:03d}"
            title = f"Sample Book {i+1}"
            author = f"Author {(i % 50) + 1}"
            genre = np.random.choice(genres, size=np.random.randint(1, 3), replace=False)
            avg_rating = np.random.normal(4.2, 0.8)
            avg_rating = max(1.0, min(5.0, avg_rating))
            
            books_data.append({
                'book_id': book_id,
                'title': title,
                'author': author,
                'genres': '|'.join(genre),
                'average_rating': avg_rating
            })
        
        books_df = pd.DataFrame(books_data)
        
        # Generate user ratings with realistic patterns
        ratings_data = []
        num_users = self.config['sample_size'] // 10  # Approximately 10 ratings per user
        
        for user_id in range(num_users):
            user_name = f"user_{user_id:04d}"
            
            # Each user has preferences for certain genres
            preferred_genres = np.random.choice(genres, size=np.random.randint(2, 4), replace=False)
            
            # Number of books this user has rated
            num_ratings = np.random.poisson(12) + 3  # Average 12 books, minimum 3
            
            # Select books with bias towards preferred genres
            user_books = []
            for _ in range(num_ratings):
                if np.random.random() < 0.7:  # 70% chance to pick from preferred genres
                    genre_books = books_df[books_df['genres'].str.contains('|'.join(preferred_genres))]
                    if len(genre_books) > 0:
                        book = genre_books.sample(1).iloc[0]
                    else:
                        book = books_df.sample(1).iloc[0]
                else:
                    book = books_df.sample(1).iloc[0]
                
                if book['book_id'] not in [b['book_id'] for b in user_books]:
                    # Generate rating with bias towards book's average rating
                    base_rating = book['average_rating']
                    user_rating = np.random.normal(base_rating, 0.5)
                    user_rating = max(1.0, min(5.0, round(user_rating * 2) / 2))  # Round to 0.5
                    
                    user_books.append({
                        'user_id': user_name,
                        'book_id': book['book_id'],
                        'rating': user_rating
                    })
            
            ratings_data.extend(user_books)
        
        ratings_df = pd.DataFrame(ratings_data)
        
        print(f"✅ Generated demo dataset:")
        print(f"   - {len(books_df)} books across {len(genres)} genres")
        print(f"   - {len(ratings_df)} ratings from {num_users} users")
        print(f"   - Average {len(ratings_df) / num_users:.1f} ratings per user")
        
        return ratings_df, books_df
    
    def demonstrate_data_loading_and_validation(self, ratings_df: pd.DataFrame, 
                                              books_df: pd.DataFrame) -> Dict[str, Any]:
        """Demonstrate data loading, validation, and exploration."""
        print("\n" + "="*60)
        print("📋 PHASE 1: DATA LOADING AND VALIDATION")
        print("="*60)
        
        # Basic dataset information
        print(f"\n📊 Dataset Overview:")
        print(f"   Ratings Dataset: {ratings_df.shape[0]:,} records, {ratings_df.shape[1]} columns")
        print(f"   Books Dataset: {books_df.shape[0]:,} records, {books_df.shape[1]} columns")
        print(f"   Unique users: {ratings_df['user_id'].nunique():,}")
        print(f"   Unique books: {ratings_df['book_id'].nunique():,}")
        
        # Data quality checks
        print(f"\n🔍 Data Quality Assessment:")
        missing_ratings = ratings_df.isnull().sum().sum()
        missing_books = books_df.isnull().sum().sum()
        print(f"   Missing values in ratings: {missing_ratings}")
        print(f"   Missing values in books: {missing_books}")
        
        # Rating distribution
        rating_stats = ratings_df['rating'].describe()
        print(f"\n⭐ Rating Distribution:")
        print(f"   Mean rating: {rating_stats['mean']:.2f}")
        print(f"   Rating range: {rating_stats['min']:.1f} - {rating_stats['max']:.1f}")
        print(f"   Ratings >= {self.config['rating_threshold']}: {(ratings_df['rating'] >= self.config['rating_threshold']).sum():,} ({(ratings_df['rating'] >= self.config['rating_threshold']).mean()*100:.1f}%)")
        
        # Genre analysis
        all_genres = []
        for genres_str in books_df['genres']:
            all_genres.extend(genres_str.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        
        print(f"\n📚 Genre Distribution (Top 5):")
        for genre, count in genre_counts.head().items():
            print(f"   {genre}: {count} books ({count/len(books_df)*100:.1f}%)")
        
        return {
            'total_ratings': len(ratings_df),
            'total_books': len(books_df),
            'unique_users': ratings_df['user_id'].nunique(),
            'unique_books': ratings_df['book_id'].nunique(),
            'avg_rating': rating_stats['mean'],
            'high_ratings_pct': (ratings_df['rating'] >= self.config['rating_threshold']).mean() * 100,
            'top_genres': genre_counts.head().to_dict(),
            'data_quality_score': 100 - (missing_ratings + missing_books) / (len(ratings_df) + len(books_df)) * 100
        }
    
    def demonstrate_preprocessing_and_basket_creation(self, ratings_df: pd.DataFrame, 
                                                    books_df: pd.DataFrame) -> Dict[str, Any]:
        """Demonstrate data preprocessing and basket creation."""
        print("\n" + "="*60)
        print("🔄 PHASE 2: DATA PREPROCESSING AND BASKET CREATION")
        print("="*60)
        
        # Merge ratings with book metadata
        print(f"\n🔗 Merging ratings with book metadata...")
        merged_df = ratings_df.merge(books_df, on='book_id', how='left')
        print(f"   Merged dataset: {len(merged_df):,} records")
        
        # Filter by rating threshold
        print(f"\n⭐ Filtering by rating threshold (>= {self.config['rating_threshold']})...")
        high_rated_df = merged_df[merged_df['rating'] >= self.config['rating_threshold']]
        print(f"   High-rated interactions: {len(high_rated_df):,} ({len(high_rated_df)/len(merged_df)*100:.1f}%)")
        
        # Create user baskets
        print(f"\n🛒 Creating user baskets...")
        user_baskets = {}
        basket_stats = []
        
        for user_id, user_data in high_rated_df.groupby('user_id'):
            book_ids = user_data['book_id'].tolist()
            ratings = user_data['rating'].tolist()
            genres = []
            for genres_str in user_data['genres']:
                genres.extend(genres_str.split('|'))
            
            user_baskets[user_id] = {
                'book_ids': book_ids,
                'ratings': ratings,
                'genres': list(set(genres)),
                'basket_size': len(book_ids)
            }
            
            basket_stats.append(len(book_ids))
        
        # Basket statistics
        basket_stats = np.array(basket_stats)
        print(f"   Total baskets created: {len(user_baskets):,}")
        print(f"   Basket size statistics:")
        print(f"     - Mean: {basket_stats.mean():.1f} books")
        print(f"     - Median: {np.median(basket_stats):.1f} books")
        print(f"     - Min: {basket_stats.min()} books")
        print(f"     - Max: {basket_stats.max()} books")
        
        # Filter baskets with minimum size
        min_basket_size = 2
        filtered_baskets = {k: v for k, v in user_baskets.items() 
                          if v['basket_size'] >= min_basket_size}
        
        print(f"\n🔍 Filtering baskets (minimum {min_basket_size} books):")
        print(f"   Baskets after filtering: {len(filtered_baskets):,}")
        print(f"   Filtered out: {len(user_baskets) - len(filtered_baskets):,} baskets")
        
        # Prepare transaction data
        print(f"\n📊 Preparing transaction matrix...")
        transactions = [basket['book_ids'] for basket in filtered_baskets.values()]
        
        # Get unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        print(f"   Total transactions: {len(transactions):,}")
        print(f"   Unique items: {len(all_items):,}")
        print(f"   Matrix size: {len(transactions):,} × {len(all_items):,}")
        print(f"   Estimated memory: {len(transactions) * len(all_items) / 1024 / 1024:.1f} MB")
        
        return {
            'total_baskets': len(user_baskets),
            'filtered_baskets': len(filtered_baskets),
            'avg_basket_size': basket_stats.mean(),
            'median_basket_size': np.median(basket_stats),
            'min_basket_size': basket_stats.min(),
            'max_basket_size': basket_stats.max(),
            'unique_items': len(all_items),
            'transactions': transactions,
            'user_baskets': filtered_baskets,
            'matrix_size': (len(transactions), len(all_items))
        }
    
    def demonstrate_frequent_itemset_mining(self, transactions: List[List[str]]) -> Dict[str, Any]:
        """Demonstrate frequent itemset mining with Apriori algorithm."""
        print("\n" + "="*60)
        print("⛏️  PHASE 3: FREQUENT ITEMSET MINING")
        print("="*60)
        
        start_time = time.time()
        
        # Create transaction matrix (simplified for demo)
        print(f"\n🔄 Creating transaction matrix...")
        from sklearn.preprocessing import MultiLabelBinarizer
        
        mlb = MultiLabelBinarizer()
        transaction_matrix = mlb.fit_transform(transactions)
        transaction_df = pd.DataFrame(transaction_matrix, columns=mlb.classes_)
        
        print(f"   Transaction matrix shape: {transaction_df.shape}")
        print(f"   Sparsity: {(transaction_df == 0).sum().sum() / (transaction_df.shape[0] * transaction_df.shape[1]) * 100:.1f}%")
        
        # Apply Apriori algorithm (simplified implementation for demo)
        print(f"\n⚡ Mining frequent itemsets (min_support = {self.config['min_support']})...")
        
        # Calculate item frequencies
        item_support = transaction_df.mean().sort_values(ascending=False)
        frequent_items = item_support[item_support >= self.config['min_support']]
        
        print(f"   Items meeting minimum support: {len(frequent_items)}")
        
        # Generate frequent itemsets (1-itemsets and 2-itemsets for demo)
        frequent_itemsets = []
        
        # 1-itemsets
        for item, support in frequent_items.items():
            frequent_itemsets.append({
                'itemset': frozenset([item]),
                'support': support,
                'size': 1
            })
        
        # 2-itemsets (simplified)
        print(f"   Generating 2-itemsets...")
        frequent_items_list = frequent_items.index.tolist()
        
        for i in range(len(frequent_items_list)):
            for j in range(i + 1, min(i + 20, len(frequent_items_list))):  # Limit for demo
                item1, item2 = frequent_items_list[i], frequent_items_list[j]
                
                # Calculate support for pair
                pair_support = ((transaction_df[item1] == 1) & (transaction_df[item2] == 1)).mean()
                
                if pair_support >= self.config['min_support']:
                    frequent_itemsets.append({
                        'itemset': frozenset([item1, item2]),
                        'support': pair_support,
                        'size': 2
                    })
        
        # Convert to DataFrame
        frequent_itemsets_df = pd.DataFrame(frequent_itemsets)
        frequent_itemsets_df = frequent_itemsets_df.sort_values('support', ascending=False)
        
        mining_time = time.time() - start_time
        
        print(f"\n✅ Mining completed in {mining_time:.2f} seconds")
        print(f"   Total frequent itemsets found: {len(frequent_itemsets_df)}")
        print(f"   1-itemsets: {(frequent_itemsets_df['size'] == 1).sum()}")
        print(f"   2-itemsets: {(frequent_itemsets_df['size'] == 2).sum()}")
        
        # Display top itemsets
        print(f"\n🏆 Top 10 Frequent Itemsets:")
        for idx, row in frequent_itemsets_df.head(10).iterrows():
            itemset_str = ', '.join(list(row['itemset']))
            print(f"   {itemset_str[:50]:<50} | Support: {row['support']:.3f}")
        
        return {
            'total_itemsets': len(frequent_itemsets_df),
            'itemsets_by_size': frequent_itemsets_df['size'].value_counts().to_dict(),
            'top_itemsets': frequent_itemsets_df.head(10).to_dict('records'),
            'mining_time': mining_time,
            'avg_support': frequent_itemsets_df['support'].mean(),
            'frequent_itemsets_df': frequent_itemsets_df
        }
    
    def demonstrate_association_rule_generation(self, frequent_itemsets_df: pd.DataFrame) -> Dict[str, Any]:
        """Demonstrate association rule generation and analysis."""
        print("\n" + "="*60)
        print("🔗 PHASE 4: ASSOCIATION RULE GENERATION")
        print("="*60)
        
        start_time = time.time()
        
        # Generate association rules from 2-itemsets
        print(f"\n⚡ Generating association rules (min_confidence = {self.config['min_confidence']})...")
        
        rules = []
        two_itemsets = frequent_itemsets_df[frequent_itemsets_df['size'] == 2]
        
        for idx, row in two_itemsets.iterrows():
            itemset = row['itemset']
            itemset_support = row['support']
            
            # Generate rules A -> B and B -> A
            items = list(itemset)
            if len(items) == 2:
                item_a, item_b = items
                
                # Find support of individual items
                item_a_support = frequent_itemsets_df[
                    frequent_itemsets_df['itemset'] == frozenset([item_a])
                ]['support'].iloc[0] if len(frequent_itemsets_df[
                    frequent_itemsets_df['itemset'] == frozenset([item_a])
                ]) > 0 else 0.01
                
                item_b_support = frequent_itemsets_df[
                    frequent_itemsets_df['itemset'] == frozenset([item_b])
                ]['support'].iloc[0] if len(frequent_itemsets_df[
                    frequent_itemsets_df['itemset'] == frozenset([item_b])
                ]) > 0 else 0.01
                
                # Rule A -> B
                confidence_ab = itemset_support / item_a_support if item_a_support > 0 else 0
                lift_ab = confidence_ab / item_b_support if item_b_support > 0 else 0
                
                if confidence_ab >= self.config['min_confidence']:
                    rules.append({
                        'antecedent': frozenset([item_a]),
                        'consequent': frozenset([item_b]),
                        'support': itemset_support,
                        'confidence': confidence_ab,
                        'lift': lift_ab
                    })
                
                # Rule B -> A
                confidence_ba = itemset_support / item_b_support if item_b_support > 0 else 0
                lift_ba = confidence_ba / item_a_support if item_a_support > 0 else 0
                
                if confidence_ba >= self.config['min_confidence']:
                    rules.append({
                        'antecedent': frozenset([item_b]),
                        'consequent': frozenset([item_a]),
                        'support': itemset_support,
                        'confidence': confidence_ba,
                        'lift': lift_ba
                    })
        
        rules_df = pd.DataFrame(rules)
        if len(rules_df) > 0:
            rules_df = rules_df.sort_values('lift', ascending=False)
        
        rule_generation_time = time.time() - start_time
        
        print(f"\n✅ Rule generation completed in {rule_generation_time:.2f} seconds")
        print(f"   Total association rules: {len(rules_df)}")
        
        if len(rules_df) > 0:
            print(f"   Average confidence: {rules_df['confidence'].mean():.3f}")
            print(f"   Average lift: {rules_df['lift'].mean():.3f}")
            print(f"   Rules with lift > 1: {(rules_df['lift'] > 1).sum()} ({(rules_df['lift'] > 1).mean()*100:.1f}%)")
            
            # Display top rules
            print(f"\n🏆 Top 10 Association Rules (by lift):")
            for idx, row in rules_df.head(10).iterrows():
                ant_str = ', '.join(list(row['antecedent']))
                con_str = ', '.join(list(row['consequent']))
                print(f"   {ant_str[:20]:<20} → {con_str[:20]:<20} | Conf: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")
        
        return {
            'total_rules': len(rules_df),
            'avg_confidence': rules_df['confidence'].mean() if len(rules_df) > 0 else 0,
            'avg_lift': rules_df['lift'].mean() if len(rules_df) > 0 else 0,
            'positive_lift_rules': (rules_df['lift'] > 1).sum() if len(rules_df) > 0 else 0,
            'rule_generation_time': rule_generation_time,
            'top_rules': rules_df.head(10).to_dict('records') if len(rules_df) > 0 else [],
            'rules_df': rules_df
        }
    
    def demonstrate_recommendation_generation(self, user_baskets: Dict, rules_df: pd.DataFrame, 
                                           books_df: pd.DataFrame) -> Dict[str, Any]:
        """Demonstrate personalized recommendation generation."""
        print("\n" + "="*60)
        print("🎯 PHASE 5: PERSONALIZED RECOMMENDATION GENERATION")
        print("="*60)
        
        recommendation_results = {}
        
        # Select demo users
        demo_users = list(user_baskets.keys())[:3]
        
        for user_id in demo_users:
            print(f"\n👤 Generating recommendations for {user_id}:")
            
            user_books = set(user_baskets[user_id]['book_ids'])
            user_genres = user_baskets[user_id]['genres']
            
            print(f"   User profile: {len(user_books)} books read")
            print(f"   Preferred genres: {', '.join(user_genres[:3])}")
            
            # Association rule-based recommendations
            rule_recommendations = []
            if len(rules_df) > 0:
                for _, rule in rules_df.iterrows():
                    antecedent = set(rule['antecedent'])
                    consequent = set(rule['consequent'])
                    
                    # Check if user has books in antecedent and doesn't have consequent
                    if antecedent.issubset(user_books) and not consequent.intersection(user_books):
                        for book_id in consequent:
                            if book_id in books_df['book_id'].values:
                                book_info = books_df[books_df['book_id'] == book_id].iloc[0]
                                rule_recommendations.append({
                                    'book_id': book_id,
                                    'title': book_info['title'],
                                    'confidence': rule['confidence'],
                                    'lift': rule['lift'],
                                    'type': 'association_rule',
                                    'explanation': f"Users who read {', '.join(antecedent)} also read this book"
                                })
            
            # Genre-based recommendations (collaborative filtering simulation)
            genre_recommendations = []
            user_genre_set = set(user_genres)
            
            for _, book in books_df.iterrows():
                if book['book_id'] not in user_books:
                    book_genres = set(book['genres'].split('|'))
                    genre_overlap = len(user_genre_set.intersection(book_genres))
                    
                    if genre_overlap > 0:
                        similarity_score = genre_overlap / len(user_genre_set.union(book_genres))
                        genre_recommendations.append({
                            'book_id': book['book_id'],
                            'title': book['title'],
                            'similarity': similarity_score,
                            'rating': book['average_rating'],
                            'type': 'genre_based',
                            'explanation': f"Based on your interest in {', '.join(book_genres.intersection(user_genre_set))}"
                        })
            
            # Sort and combine recommendations
            rule_recommendations = sorted(rule_recommendations, key=lambda x: x['lift'], reverse=True)[:5]
            genre_recommendations = sorted(genre_recommendations, key=lambda x: x['similarity'] * x['rating'], reverse=True)[:5]
            
            all_recommendations = rule_recommendations + genre_recommendations
            
            print(f"   Association rule recommendations: {len(rule_recommendations)}")
            print(f"   Genre-based recommendations: {len(genre_recommendations)}")
            
            # Display top recommendations
            print(f"   Top 5 recommendations:")
            for i, rec in enumerate(all_recommendations[:5], 1):
                if rec['type'] == 'association_rule':
                    score_str = f"Lift: {rec['lift']:.2f}"
                else:
                    score_str = f"Score: {rec['similarity'] * rec['rating']:.2f}"
                print(f"     {i}. {rec['title'][:40]:<40} | {score_str} | {rec['type']}")
            
            recommendation_results[user_id] = {
                'user_books_count': len(user_books),
                'user_genres': user_genres,
                'rule_recommendations': rule_recommendations,
                'genre_recommendations': genre_recommendations,
                'total_recommendations': len(all_recommendations)
            }
        
        return {
            'demo_users': demo_users,
            'user_results': recommendation_results,
            'avg_recommendations_per_user': np.mean([r['total_recommendations'] for r in recommendation_results.values()]),
            'recommendation_types': {
                'association_rule': sum(len(r['rule_recommendations']) for r in recommendation_results.values()),
                'genre_based': sum(len(r['genre_recommendations']) for r in recommendation_results.values())
            }
        }
    
    def analyze_business_insights(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate business insights from the analysis results."""
        print("\n" + "="*60)
        print("💡 PHASE 6: BUSINESS INSIGHTS AND ANALYSIS")
        print("="*60)
        
        insights = []
        
        # Dataset insights
        dataset_stats = all_results['dataset_stats']
        insights.append(f"Dataset Quality: {dataset_stats['data_quality_score']:.1f}% - High quality data with minimal missing values")
        insights.append(f"User Engagement: {dataset_stats['high_ratings_pct']:.1f}% of ratings are high (≥{self.config['rating_threshold']}), indicating strong user satisfaction")
        
        # Preprocessing insights
        preprocessing_stats = all_results['preprocessing_stats']
        avg_basket_size = preprocessing_stats['avg_basket_size']
        if avg_basket_size > 10:
            insights.append(f"High User Engagement: Average basket size of {avg_basket_size:.1f} books indicates active readers")
        elif avg_basket_size > 5:
            insights.append(f"Moderate User Engagement: Average basket size of {avg_basket_size:.1f} books shows regular reading habits")
        else:
            insights.append(f"Low User Engagement: Average basket size of {avg_basket_size:.1f} books suggests casual readers")
        
        # Mining insights
        mining_results = all_results['mining_results']
        total_itemsets = mining_results['total_itemsets']
        if total_itemsets > 100:
            insights.append(f"Rich Pattern Discovery: {total_itemsets} frequent itemsets found, indicating strong book co-occurrence patterns")
        else:
            insights.append(f"Limited Pattern Discovery: {total_itemsets} frequent itemsets found, may need to adjust minimum support threshold")
        
        # Association rule insights
        rule_results = all_results.get('rule_results', {})
        if rule_results:
            positive_lift_pct = (rule_results['positive_lift_rules'] / max(1, rule_results['total_rules'])) * 100
            if positive_lift_pct > 70:
                insights.append(f"Strong Associations: {positive_lift_pct:.1f}% of rules have positive lift, indicating meaningful book relationships")
            else:
                insights.append(f"Weak Associations: {positive_lift_pct:.1f}% of rules have positive lift, suggesting limited cross-selling opportunities")
        
        # Recommendation insights
        rec_results = all_results.get('recommendation_results', {})
        if rec_results:
            avg_recs = rec_results['avg_recommendations_per_user']
            if avg_recs > 8:
                insights.append(f"High Recommendation Coverage: Average {avg_recs:.1f} recommendations per user enables personalized experiences")
            else:
                insights.append(f"Limited Recommendation Coverage: Average {avg_recs:.1f} recommendations per user may require algorithm tuning")
        
        # Genre insights
        top_genres = dataset_stats['top_genres']
        most_popular_genre = list(top_genres.keys())[0]
        insights.append(f"Genre Preference: '{most_popular_genre}' is the most popular genre, representing {list(top_genres.values())[0]} books")
        
        # Performance insights
        if 'mining_time' in mining_results:
            mining_time = mining_results['mining_time']
            if mining_time < 5:
                insights.append(f"Efficient Processing: Mining completed in {mining_time:.1f}s, suitable for real-time applications")
            else:
                insights.append(f"Processing Time: Mining took {mining_time:.1f}s, consider optimization for production use")
        
        # Business recommendations
        insights.append("Business Recommendation: Implement cross-selling campaigns based on discovered association rules")
        insights.append("Business Recommendation: Use genre-based recommendations for new user onboarding")
        insights.append("Business Recommendation: Monitor recommendation click-through rates to optimize algorithms")
        
        print(f"\n🎯 Key Business Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
    
    def run_complete_demonstration(self) -> DemoResults:
        """Run the complete end-to-end demonstration."""
        print("🚀 STARTING COMPLETE MARKET BASKET ANALYSIS DEMONSTRATION")
        print("="*80)
        
        # Generate demo data
        ratings_df, books_df = self.generate_demo_data()
        
        # Phase 1: Data loading and validation
        dataset_stats = self.demonstrate_data_loading_and_validation(ratings_df, books_df)
        
        # Phase 2: Preprocessing and basket creation
        preprocessing_stats = self.demonstrate_preprocessing_and_basket_creation(ratings_df, books_df)
        
        # Phase 3: Frequent itemset mining
        mining_results = self.demonstrate_frequent_itemset_mining(preprocessing_stats['transactions'])
        
        # Phase 4: Association rule generation
        rule_results = self.demonstrate_association_rule_generation(mining_results['frequent_itemsets_df'])
        
        # Phase 5: Recommendation generation
        recommendation_results = self.demonstrate_recommendation_generation(
            preprocessing_stats['user_baskets'], 
            rule_results['rules_df'], 
            books_df
        )
        
        # Combine all results
        all_results = {
            'dataset_stats': dataset_stats,
            'preprocessing_stats': preprocessing_stats,
            'mining_results': mining_results,
            'rule_results': rule_results,
            'recommendation_results': recommendation_results
        }
        
        # Phase 6: Business insights
        business_insights = self.analyze_business_insights(all_results)
        
        # Performance metrics
        performance_metrics = {
            'total_execution_time': mining_results.get('mining_time', 0) + rule_results.get('rule_generation_time', 0),
            'data_processing_efficiency': len(preprocessing_stats['transactions']) / max(1, mining_results.get('mining_time', 1)),
            'recommendation_coverage': recommendation_results['avg_recommendations_per_user'],
            'pattern_discovery_rate': mining_results['total_itemsets'] / len(preprocessing_stats['transactions']) * 100
        }
        
        # Create final results
        self.results = DemoResults(
            dataset_stats=dataset_stats,
            preprocessing_stats=preprocessing_stats,
            mining_results=mining_results,
            recommendation_results=recommendation_results,
            business_insights=business_insights,
            performance_metrics=performance_metrics
        )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\n📊 Executive Summary:")
        print(f"   • Processed {dataset_stats['total_ratings']:,} ratings from {dataset_stats['unique_users']:,} users")
        print(f"   • Discovered {mining_results['total_itemsets']} frequent itemsets and {rule_results['total_rules']} association rules")
        print(f"   • Generated {recommendation_results['avg_recommendations_per_user']:.1f} recommendations per user on average")
        print(f"   • Identified {len(business_insights)} key business insights")
        print(f"   • Total processing time: {performance_metrics['total_execution_time']:.2f} seconds")
        
        return self.results

if __name__ == "__main__":
    # Run the complete demonstration
    demo = MarketBasketAnalysisDemo()
    results = demo.run_complete_demonstration()
    
    # Save results for further analysis
    with open('demo_results.json', 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'dataset_stats': results.dataset_stats,
            'preprocessing_stats': {k: v for k, v in results.preprocessing_stats.items() 
                                  if k not in ['transactions', 'user_baskets']},
            'mining_results': {k: v for k, v in results.mining_results.items() 
                             if k != 'frequent_itemsets_df'},
            'business_insights': results.business_insights,
            'performance_metrics': results.performance_metrics
        }
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to 'demo_results.json'")