import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, Image, Alert, Dimensions } from 'react-native';
import { getJobResults } from '../api/feedlyApi';
import { commonStyles as cs } from './_styles';

const { width } = Dimensions.get('window');

export default function ResultsScreen({ route, navigation }) {
  const { jobId } = route.params;
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (!jobId) {
      Alert.alert("오류", "Job ID가 없습니다.", [{ text: "OK", onPress: () => navigation.goBack() }]);
      return;
    }

    const fetchResults = async () => {
      try {
        const data = await getJobResults(jobId);
        setResults(data);
      } catch (error) {
        Alert.alert("결과 로딩 실패", error.message, [{ text: "OK", onPress: () => navigation.goBack() }]);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [jobId]);

  if (loading) {
    return (
      <View style={[cs.container, styles.center]}>
        <ActivityIndicator size="large" color="#4f46e5" />
        <Text style={styles.loadingText}>결과를 불러오는 중...</Text>
      </View>
    );
  }

  if (!results) {
    return (
      <View style={[cs.container, styles.center]}>
        <Text>결과를 찾을 수 없습니다.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={cs.container}>
      <View style={styles.header}>
        <Text style={styles.title}>생성된 광고 시안</Text>
      </View>

      {/* Image Carousel/List */}
      <ScrollView horizontal pagingEnabled showsHorizontalScrollIndicator={false} style={styles.carousel}>
        {results.images.map((img, index) => (
          <Image key={index} source={{ uri: img.image_url }} style={styles.image} resizeMode="cover" />
        ))}
      </ScrollView>

      {/* Ad Copy Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>광고 문구</Text>
        {/* ENSURE this uses 'instagram_ad_copy' to match the backend fix */}
        <Text style={styles.cardContent}>{results.instagram_ad_copy}</Text>
      </View>

      {/* Hashtags Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>해시태그</Text>
        <Text style={styles.cardContent}>{results.hashtags}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { justifyContent: 'center', alignItems: 'center' },
  loadingText: { marginTop: 10, fontSize: 16, color: '#6b7280' },
  header: { padding: 20 },
  title: { fontSize: 24, fontWeight: 'bold' },
  carousel: { height: width }, // Makes carousel square
  image: { width: width, height: width },
  card: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginHorizontal: 20,
    marginTop: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
    elevation: 3,
  },
  cardTitle: { fontSize: 18, fontWeight: 'bold', marginBottom: 8 },
  cardContent: { fontSize: 16, lineHeight: 24, color: '#374151' },
});