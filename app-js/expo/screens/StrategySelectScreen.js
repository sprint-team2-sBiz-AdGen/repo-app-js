import React, { useState } from "react";
import { View, Text, TouchableOpacity, FlatList, StyleSheet } from "react-native";
import { commonStyles as cs } from "./_styles";

const STRATEGIES = [
  { id: "hero", label: "Hero Dish", hint: "음식 접사를 올리면 좋아요." },
  { id: "seasonal", label: "Seasonal Limited", hint: "제철 메뉴, 한정 메뉴에 잘 어울립니다." },
  // { id: "bts", label: "Behind-the-Scenes", hint: "조리 과정, 주방, 손이 나오는 사진이 좋아요." },
  // { id: "lifestyle", label: "Lifestyle", hint: "테이블, 손, 분위기가 보이는 사진을 추천합니다." },
  // { id: "ugc", label: "UGC / Social Proof", hint: "손님이 찍어준 느낌의 사진이 어울려요." },
  // { id: "minimal", label: "Minimalist", hint: "깔끔한 배경, 심플한 구성의 사진이 좋습니다." },
  { id: "comfort", label: "Emotion / Cozy", hint: "따뜻한 조명, 집밥 느낌 사진에 어울립니다." },
  { id: "retro", label: "Retro / Vintage", hint: "가게 전경, 오래된 간판, 분위기 있는 사진 추천." },
];

export default function StrategySelectScreen({ navigation }) {
  const [selected, setSelected] = useState(null);

  const goNext = () => {
    if (!selected) return;
    navigation.navigate("PhotoAndDescription", { strategy: selected });
  };

  return (
    <View style={cs.container}>
      <Text style={cs.title}>1단계. 광고 스타일을 골라주세요</Text>
      <Text style={cs.subtitle}>
        스타일을 먼저 고르면, 어떤 사진과 설명을 넣어야 할지 더 쉽게 떠오릅니다.
      </Text>

      <FlatList
        data={STRATEGIES}
        keyExtractor={(item) => item.id}
        numColumns={2}
        columnWrapperStyle={{ gap: 10 }}
        contentContainerStyle={{ gap: 10, paddingBottom: 20 }}
        renderItem={({ item }) => {
          const active = selected?.id === item.id;
          return (
            <TouchableOpacity
              onPress={() => setSelected(item)}
              style={[
                styles.card,
                active && { borderColor: "#111827", backgroundColor: "#f3f4f6" },
              ]}
            >
              <Text style={styles.cardTitle}>{item.label}</Text>
              <Text style={styles.cardHint}>{item.hint}</Text>
            </TouchableOpacity>
          );
        }}
      />

      <TouchableOpacity
        style={[
          cs.primaryButton,
          !selected && { backgroundColor: "#d1d5db" },
        ]}
        disabled={!selected}
        onPress={goNext}
      >
        <Text style={cs.primaryButtonText}>다음 (사진 & 설명)</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    padding: 12,
  },
  cardTitle: {
    fontWeight: "600",
    marginBottom: 6,
  },
  cardHint: {
    fontSize: 11,
    color: "#6b7280",
  },
});
