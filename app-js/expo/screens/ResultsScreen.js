// screens/ResultsScreen.js
import React from "react";
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from "react-native";
import { commonStyles as cs } from "./_styles";

export default function ResultsScreen({ route, navigation }) {
  const { results = [], strategy, description } = route.params || {};

  const handleSelect = (item) => {
    navigation.navigate("Share", { selected: item, strategy, description });
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity style={styles.card} onPress={() => handleSelect(item)}>
      <Text style={styles.cardTitle}>{item.title}</Text>
      <Text style={styles.cardCaption}>{item.copy_ko}</Text>
      <Text style={styles.selectText}>이 광고 선택하기</Text>
    </TouchableOpacity>
  );

  return (
    <View style={cs.container}>
      <Text style={cs.title}>광고 결과</Text>
      <Text style={cs.subtitle}>
        아래 후보 중에서 마음에 드는 광고를 선택해 보세요.
      </Text>

      <FlatList
        data={results}
        keyExtractor={(item) => String(item.id)}
        renderItem={renderItem}
        contentContainerStyle={{ paddingVertical: 16 }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "white",
    borderRadius: 16,
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
    marginBottom: 8,
  },
  selectText: {
    fontSize: 12,
    color: "#111827",
    fontWeight: "600",
  },
});