import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Alert, Image } from "react-native";
import * as Clipboard from "expo-clipboard";
import { commonStyles as cs } from "./_styles";
import ViewShot from "react-native-view-shot";

export default function ShareScreen({ route, navigation }) {
  const { selected, description, strategy } = route.params || {};

  const fullCaption = `${selected?.caption || ""}\n\n${description || ""}\n\n#feedlyai #맛집`;

  const copyToClipboard = async () => {
    await Clipboard.setStringAsync(fullCaption);
    Alert.alert("복사 완료", "인스타그램에 붙여넣기 하시면 됩니다.");
  };

  return (
    <View style={cs.container}>
      <Text style={cs.title}>광고 & 인스타그램 피드</Text>
      <Text style={cs.subtitle}>
        아래 내용을 복사해서 인스타그램 글쓰기 창에 붙여넣으시면 됩니다.
      </Text>

      <View style={styles.box}>
        <Text style={styles.label}>스타일</Text>
        <Text style={styles.value}>{strategy?.label}</Text>

        <Text style={[styles.label, { marginTop: 12 }]}>최종 광고</Text>
        <Text style={styles.value}>{fullCaption}</Text>
      </View>

      <ViewShot style={styles.imageContainer}>
        <Image source={{ uri: selected?.imageUri }} style={styles.image} />
        <Text style={styles.overlayText}>{selected?.caption}</Text>
      </ViewShot>

      <TouchableOpacity style={cs.primaryButton} onPress={copyToClipboard}>
        <Text style={cs.primaryButtonText}>복사하기</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[cs.mutedButton, { marginTop: 12 }]}
        onPress={() => navigation.navigate("Home")}
      >
        <Text style={cs.mutedButtonText}>홈으로 돌아가기</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    padding: 12,
    backgroundColor: "white",
    marginBottom: 16,
  },
  label: {
    fontSize: 12,
    color: "#6b7280",
    marginBottom: 2,
  },
  value: {
    fontSize: 13,
    color: "#111827",
  },
  imageContainer: {
    width: "100%",
    height: 300,
    marginTop: 16,
    borderRadius: 12,
    overflow: "hidden",
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },
  overlayText: {
    position: "absolute",
    bottom: 20,
    left: 20,
    color: "white",
    fontSize: 24,
  },
});
