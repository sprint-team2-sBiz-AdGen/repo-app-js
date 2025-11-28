import React, { useState, useEffect } from "react";
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, ActivityIndicator } from "react-native";
import { commonStyles as cs } from "./_styles";
import { getGenerationById } from "../api/feedlyApi";
import { STRATEGIES } from "../constants/strategies";

export default function ResultsScreen({ route, navigation }) {
  // We need the original imageUri and description, so we'll get them from route.params
  const { generationId, imageUri, description } = route.params;
  const [isLoading, setIsLoading] = useState(true);
  const [generation, setGeneration] = useState(null);

  useEffect(() => {
    const loadResult = async () => {
      if (!generationId) return;
      try {
        setIsLoading(true);
        const data = await getGenerationById(generationId);
        setGeneration(data);
      } catch (error) {
        console.error("Failed to load result:", error);
      } finally {
        setIsLoading(false);
      }
    };
    loadResult();
  }, [generationId]);

  // --- FIX 1: Create the function to handle navigation ---
  const onSelectVariant = (variant) => {
    const strategy = STRATEGIES.find(s => s.name === generation.strategy_name);

    // Navigate to ShareScreen with all the required data
    navigation.navigate("Share", {
      selected: {
        caption: variant.copy_ko,
        imageUri: imageUri, // Pass the original image URI
      },
      description: description, // Pass the original description
      strategy: strategy,
    });
  };

  if (isLoading) {
    return (
      <View style={[cs.container, { justifyContent: "center" }]}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!generation) {
    return (
      <View style={cs.container}>
        <Text>결과를 불러오지 못했습니다.</Text>
      </View>
    );
  }

  const strategy = STRATEGIES.find(s => s.name === generation.strategy_name);
  const strategyLabel = strategy ? strategy.label : generation.strategy_name;

  return (
    <ScrollView style={cs.container}>
      <Text style={cs.title}>생성된 광고</Text>
      <Text style={cs.subtitle}>스타일: {strategyLabel}</Text>

      <View style={styles.container}>
        {generation.variants.map((variant, index) => (
          // --- FIX 2: Add the onPress handler to the TouchableOpacity ---
          <TouchableOpacity
            key={index}
            style={styles.variantCard}
            onPress={() => onSelectVariant(variant)}
          >
            <Text style={styles.cardTitle}>후보 {index + 1}</Text>
            <Text style={styles.cardCaption}>{variant.copy_ko}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 16,
  },
  variantCard: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  cardTitle: {
    fontSize: 13,
    color: "#6b7280",
    fontWeight: "600",
    marginBottom: 4,
  },
  cardCaption: {
    fontSize: 14,
    color: "#111827",
  },
});