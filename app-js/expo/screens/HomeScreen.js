import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import { commonStyles as cs } from "./_styles";

export default function HomeScreen({ navigation }) {
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
