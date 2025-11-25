import React, { useEffect, useState } from "react";
import { View, Text, ActivityIndicator, StyleSheet, Alert } from "react-native";
import { translateDescription, generateCopyVariants, uploadImage } from "../api/feedlyApi";
import { getStrategyMeta } from "../utils/strategyMap";

export default function GeneratingScreen({ route, navigation }) {
  const { strategy, imageUri, description } = route.params || {};
  const [statusMessage, setStatusMessage] = useState("설명을 분석하고 있어요...");

  useEffect(() => {
    const run = async () => {
      try {
        if (!description || !strategy) {
          Alert.alert("오류", "전달된 정보가 부족합니다. 처음부터 다시 시도해 주세요.");
          navigation.goBack();
          return;
        }

        // 0) Upload image if provided
        let imageId = null;
        if (imageUri) {
        setStatusMessage("사진을 업로드하고 있어요...");
        const uploadResult = await uploadImage(imageUri);
        imageId = uploadResult.id;
        }
        
        // 1) Translate KR description → EN and save in DB
        setStatusMessage("사장님 설명을 번역하고 저장하고 있어요...");
        const translateResult = await translateDescription(description);
        const descriptionId = translateResult.id;

        // 2) Strategy mapping for backend
        const { apiId: strategyId, apiName: strategyName } = getStrategyMeta(
          strategy.id
        );

        // For now, use description as product_name (you can refine later)
        const productName = description;

        // 3) Generate 3 Korean ad copies & save to DB
        setStatusMessage("광고 문구를 생성하고 있어요...");
        const copyResult = await generateCopyVariants({
          descriptionId,
          strategyId,
          strategyName,
          productName,
          imageId,
          foregroundAnalysis: "", // later: pass LLAVA analysis if you want
        });

        const variants = copyResult.variants || [];
        if (!variants.length) {
          throw new Error("No variants returned from backend.");
        }

        const results = variants.map((v, index) => ({
          id: v.id,
          title: `제안 ${index + 1}`,
          copy_ko: v.copy_ko,
        }));

        navigation.replace("Results", {
          results,
          strategy,
          description,
        });
      } catch (err) {
        console.error(err);
        Alert.alert(
          "생성 중 오류가 발생했어요",
          err?.message || "잠시 후 다시 시도해 주세요."
        );
        navigation.goBack();
      }
    };

    run();
  }, [description, strategy, navigation]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#facc15" />
      <Text style={styles.text}>광고를 생성하는 중입니다...</Text>
      <Text style={[styles.text, { marginTop: 8, fontSize: 13 }]}>
        {statusMessage}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111827",
    alignItems: "center",
    justifyContent: "center",
  },
  text: {
    marginTop: 16,
    color: "#e5e7eb",
    fontSize: 14,
  },
});