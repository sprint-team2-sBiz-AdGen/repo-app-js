import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, Image, Dimensions } from "react-native"; // Add Image and Dimensions
import { getJobResults } from "../api/feedlyApi";
import { commonStyles as cs } from "./_styles";

// The base URL of your FastAPI server
const API_BASE_URL = 'http://34.9.178.28:8012'; 
const screenWidth = Dimensions.get('window').width;

export default function ResultsScreen({ route }) {
  const { jobId } = route.params;
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        setLoading(true);
        const data = await getJobResults(jobId);
        setResults(data);
        setError(null);
      } catch (e) {
        console.error("Failed to fetch results:", e);
        setError("결과를 불러오는 데 실패했습니다.");
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, [jobId]);

  if (loading) {
    return <View style={cs.container}><ActivityIndicator size="large" /></View>;
  }
  if (error) {
    return <View style={cs.container}><Text style={cs.text}>{error}</Text></View>;
  }
  if (!results) {
    return <View style={cs.container}><Text style={cs.text}>결과가 없습니다.</Text></View>;
  }

  return (
    <ScrollView style={cs.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.header}>생성된 광고 시안</Text>

      {/* Image Carousel Section */}
      <ScrollView horizontal pagingEnabled showsHorizontalScrollIndicator={false} style={styles.carousel}>
        {results.images && results.images.map((img, index) => {
          // FIX: Use the full URL path provided by the API directly.
          // The API now returns the correct path like "/assets/yh/tenants/...".
          // The api.js layer will prepend the base URL (http://34.9.178.28:8012).
          const imageUrl = `${API_BASE_URL}${img.image_url}`;
          console.log(`Rendering image URL: ${imageUrl}`); // Debug log
          return (
            <View key={index} style={styles.imageContainer}>
              <Image
                source={{ uri: imageUrl }}
                style={styles.image}
                resizeMode="cover"
              />
            </View>
          );
        })}
      </ScrollView>

      {/* Ad Copy Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>인스타 피드글</Text>
        <Text style={styles.cardContent}>{results.instagram_ad_copy}</Text>
      </View>

      {/* Hashtags Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>추천 해시태그</Text>
        <Text style={styles.cardContent}>{results.hashtags}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  contentContainer: {
    padding: 16,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  carousel: {
    marginBottom: 24,
  },
  imageContainer: {
    width: screenWidth - 32, // Full width with padding
    height: screenWidth - 32, // Make it square
    borderRadius: 12,
    overflow: 'hidden',
    marginRight: 16, // Space between images if you remove pagingEnabled
  },
  image: {
    width: '100%',
    height: '100%',
  },
  card: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  cardContent: {
    fontSize: 16,
    lineHeight: 24,
    color: '#495057',
  },
});