import React, { useState, useMemo } from "react";
import { View, Text, TouchableOpacity, Image, TextInput, StyleSheet, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { commonStyles as cs } from "./_styles";

const getDescriptionPlaceholder = (strategyId) => {
  switch (strategyId) {
    case "hero":
      return "예: 갓 튀긴 바삭한 치킨과 촉촉한 속살이 자랑인 시그니처 메뉴입니다.";
    case "seasonal":
      return "예: 이번 주까지만 판매하는 여름 한정 냉모밀 세트입니다.";
    case "bts":
      return "예: 매일 아침 직접 반죽하고 굽는 수제 빵입니다.";
    case "lifestyle":
      return "예: 여유로운 브런치 타임에 잘 어울리는 샌드위치와 라떼 세트입니다.";
    case "ugc":
      return "예: 손님들이 직접 찍어 올려주는 인기 메뉴입니다.";
    case "minimal":
      return "예: 재료 그대로의 맛을 살린 담백한 평양냉면입니다.";
    case "comfort":
      return "예: 집밥처럼 편안한 된장찌개 정식입니다.";
    case "retro":
      return "예: 90년대 감성을 그대로 담은 추억의 분식 세트입니다.";
    default:
      return "예: 오늘의 대표 메뉴를 한 줄로 소개해 주세요.";
  }
};

export default function PhotoAndDescriptionScreen({ route, navigation }) {
  const { strategy } = route.params || {};
  const [imageUri, setImageUri] = useState(null);
  const [description, setDescription] = useState("");

  const strategyLabel = strategy?.label || "선택한 스타일";
  const descriptionPlaceholder = useMemo(
    () => getDescriptionPlaceholder(strategy?.id),
    [strategy]
  );

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 5],
      quality: 0.8,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

  const goNext = () => {
    if (!imageUri) {
      Alert.alert("사진이 필요합니다", "대표 메뉴 사진을 하나 선택해 주세요.");
      return;
    }
    if (!description.trim()) {
      Alert.alert("한 줄 설명이 필요합니다", "메뉴나 가게를 한 줄로 소개해 주세요.");
      return;
    }
    navigation.navigate("Generating", { strategy, imageUri, description });
  };

  return (
    <View style={cs.container}>
      <Text style={cs.title}>2단계. 사진 & 설명</Text>
      <Text style={cs.subtitle}>
        선택하신 스타일: {strategyLabel}
      </Text>

      <TouchableOpacity style={styles.imagePicker} onPress={pickImage}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.image} />
        ) : (
          <Text style={{ color: "#6b7280" }}>여기를 눌러 이 스타일에 어울리는 사진을 선택하세요</Text>
        )}
      </TouchableOpacity>

      <Text style={[cs.subtitle, { marginTop: 8 }]}>스타일에 맞는 설명을 써주세요</Text>
      <TextInput
        style={styles.input}
        value={description}
        onChangeText={setDescription}
        placeholder={descriptionPlaceholder}
        multiline
      />

      <TouchableOpacity style={cs.primaryButton} onPress={goNext}>
        <Text style={cs.primaryButtonText}>광고 생성하기</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  imagePicker: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderStyle: "dashed",
    borderRadius: 16,
    height: 220,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f9fafb",
    marginBottom: 12,
  },
  image: {
    width: "100%",
    height: "100%",
    borderRadius: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 12,
    padding: 12,
    fontSize: 14,
    backgroundColor: "white",
    minHeight: 80,
    textAlignVertical: "top",
  },
});
