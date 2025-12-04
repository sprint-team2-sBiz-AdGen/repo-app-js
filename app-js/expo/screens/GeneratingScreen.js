import React, { useEffect, useState, useRef } from "react";
import { View, Text, ActivityIndicator, StyleSheet, Alert } from "react-native";
import { commonStyles as cs } from "./_styles";
// --- FIX: Import 'api' as a named export inside curly braces ---
import { api, gptKorToEng, gptAdCopyEng, gptAdCopyKor } from "../api/feedlyApi";

export default function GeneratingScreen({ route, navigation }) {
  const { jobId } = route.params;
  const [statusText, setStatusText] = useState("작업을 시작합니다...");

  // Use refs to hold timer IDs to prevent issues with stale state in closures
  const pollingIntervalRef = useRef(null);
  const timeoutRef = useRef(null);

  // Function to clear all timers
  const stopAllTimers = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };

  useEffect(() => {
    if (!jobId) {
      Alert.alert("오류", "Job ID가 없습니다.", [
        { text: "OK", onPress: () => navigation.goBack() },
      ]);
      return;
    }

    // Set a 30-minute overall timeout for the entire process
    timeoutRef.current = setTimeout(() => {
      stopAllTimers();
      Alert.alert(
        "시간 초과",
        "생성 작업이 예상보다 오래 걸리고 있습니다. 잠시 후 결과 화면에서 다시 확인해주세요.",
        [{ text: "확인", onPress: () => navigation.goBack() }]
      );
    }, 40 * 60 * 1000); // 40 minutes

    // Function to poll for results
    const startPolling = () => {
      pollingIntervalRef.current = setInterval(async () => {
        try {
          console.log(`Polling for results for job: ${jobId}`);
          // This endpoint should return the final results for the job
          const response = await api.get(`/api/v1/jobs/${jobId}/results`);
          const results = response.data;

          // Check if results are ready (e.g., has images and a valid ad copy)
          if (results && results.images && results.images.length > 0 && results.instagram_ad_copy) {
            console.log("Success: Final results are ready.");
            stopAllTimers();
            navigation.replace("ResultScreen", { jobId }); // Navigate to the final screen
          } else {
            console.log("Results not ready yet, continuing to poll...");
          }
        } catch (error) {
          // A 404 error is expected if the results aren't created yet, so we continue polling.
          if (error.response && error.response.status === 404) {
            console.log("Results not found (404), continuing to poll...");
          } else {
            // For other errors, stop the process and alert the user.
            console.error("Error during polling:", error);
            stopAllTimers();
            Alert.alert(
              "오류 발생",
              `결과를 가져오는 중 문제가 발생했습니다: ${error.message}`,
              [{ text: "OK", onPress: () => navigation.goBack() }]
            );
          }
        }
      }, 10000); // Poll every 10 seconds
    };

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
        await gptAdCopyKor(jobId);

        // Step 4: Start polling for the final results from the backend
        setStatusText("이미지와 최종 광고 문구를 생성 중입니다...");
        startPolling();

      } catch (error) {
        console.error("Failed to run generation pipeline:", error);
        stopAllTimers(); // Stop timers on failure
        Alert.alert(
          "오류 발생",
          `광고 문구를 생성하는 중 문제가 발생했습니다: ${error.message}`,
          [{ text: "OK", onPress: () => navigation.goBack() }]
        );
      }
    };

    runGenerationPipeline();

    // Cleanup function to clear timers when the component unmounts
    return () => {
      stopAllTimers();
    };
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