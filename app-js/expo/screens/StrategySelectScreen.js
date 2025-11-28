import React, { useState } from "react";
import { View, Text, TouchableOpacity, FlatList, StyleSheet } from "react-native";
import { commonStyles as cs } from "./_styles";
// --- FIX 1: Import from the new central file ---
import { STRATEGIES } from "../constants/strategies";

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
        스타일을 먼저 고르면, 어떤 사진과 설명을 넣어야할지 더 쉽게 떠오릅니다.
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
