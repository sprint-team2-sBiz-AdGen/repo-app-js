import React, { useEffect } from "react";
import { View, Text, ActivityIndicator, StyleSheet, Alert } from "react-native";
import { commonStyles as cs } from "./_styles";
import { generateCopyVariants } from "../api/feedlyApi";

export default function GeneratingScreen({ route, navigation }) {
  // Get all the params
  const { strategy, imageId, descriptionId, imageUri, description } = route.params;

  useEffect(() => {
    const generate = async () => {
      try {
        const response = await generateCopyVariants({
          strategy_id: strategy.id,
          strategy_name: strategy.name,
          image_id: imageId,
          description_id: descriptionId,
        });
        
        // Pass all necessary data to the ResultsScreen
        navigation.replace("Results", {
          generationId: response.generation_id,
          imageUri: imageUri,
          description: description,
        });

      } catch (error) {
        console.error("Failed to generate copy variants:", error);
        Alert.alert(
          "오류 발생",
          "광고 문구를 생성하는 중 문제가 발생했습니다.",
          [{ text: "OK", onPress: () => navigation.goBack() }]
        );
      }
    };
    generate();
  }, []);

  return (
    <View style={[cs.container, styles.center]}>
      <ActivityIndicator size="large" color="#4f46e5" />
      <Text style={styles.text}>최적의 광고를{"\n"}생성하고 있어요...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  text: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
    color: "#374151",
  },
});