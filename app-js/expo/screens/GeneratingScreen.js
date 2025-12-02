import React, { useEffect, useState } from "react";
import { View, Text, ActivityIndicator, StyleSheet, Alert } from "react-native";
import { commonStyles as cs } from "./_styles";
import { gptKorToEng, gptAdCopyEng, gptAdCopyKor } from "../api/feedlyApi";

export default function GeneratingScreen({ route, navigation }) {
  // The 'jobId' is now passed from the previous screen
  const { jobId } = route.params;
  const [statusText, setStatusText] = useState("작업을 시작합니다...");

  useEffect(() => {
    if (!jobId) {
      Alert.alert("오류", "Job ID가 없습니다.", [
        { text: "OK", onPress: () => navigation.goBack() },
      ]);
      return;
    }

    const runGenerationPipeline = async () => {
      try {
        // Step 1: Translate Korean description to English
        setStatusText("설명을 번역하는 중...");
        await gptKorToEng(jobId);

        // Step 2: Generate English ad copy
        setStatusText("영문 광고 문구를 생성하는 중...");
        await gptAdCopyEng(jobId);

        // Step 3: Translate English ad copy back to Korean
        setStatusText("한글 광고 문구로 최종 번역 중...");
        const finalResponse = await gptAdCopyKor(jobId);

        // Navigate to Results screen with the completed job ID
        // The Results screen will fetch the final data using this ID.
        navigation.replace("Results", {
          jobId: jobId,
          // Pass along any other necessary data if available in finalResponse
        });
      } catch (error) {
        console.error("Failed to run generation pipeline:", error);
        Alert.alert(
          "오류 발생",
          `광고 문구를 생성하는 중 문제가 발생했습니다: ${error.message}`,
          [{ text: "OK", onPress: () => navigation.goBack() }]
        );
      }
    };

    runGenerationPipeline();
  }, [jobId, navigation]);

  return (
    <View style={[cs.container, styles.center]}>
      <ActivityIndicator size="large" color="#4f46e5" />
      <Text style={styles.text}>최적의 광고를{"\n"}생성하고 있어요...</Text>
      <Text style={styles.statusText}>{statusText}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f9fafb",
  },
  text: {
    marginTop: 20,
    fontSize: 22,
    fontWeight: "bold",
    textAlign: "center",
    color: "#1f2937",
  },
  statusText: {
    marginTop: 15,
    fontSize: 16,
    color: "#6b7280",
  },
});