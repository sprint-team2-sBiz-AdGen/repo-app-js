import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import { commonStyles as cs } from "./_styles";
import * as ImagePicker from "expo-image-picker";

export default function HomeScreen({ navigation }) {
  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      // --- FIX: Change MediaTypeOptions to MediaType ---
      mediaTypes: ImagePicker.MediaType.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
  };

  return (
    <View style={cs.container}>
      <Text style={cs.title}>안녕하세요, 사장님 👋</Text>
      <Text style={cs.subtitle}>
        먼저 원하는 스타일을 고른 다음, 사진과 설명만 넣으면 끝입니다.
      </Text>

      <TouchableOpacity
        style={cs.primaryButton}
        onPress={() => navigation.navigate("StrategySelect")}
      >
        <Text style={cs.primaryButtonText}>1단계 시작하기 (광고 스타일 선택)</Text>
      </TouchableOpacity>
    </View>
  );
}
