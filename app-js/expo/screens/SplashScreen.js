import React, { useEffect } from "react";
import { View, Text, StyleSheet } from "react-native";

export default function SplashScreen({ navigation }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace("Home");
    }, 1200);
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <Text style={styles.logo}>Feedly AI</Text>
      <Text style={styles.tagline}>사진 한 장으로 만드는 인스타 광고</Text>
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
  logo: {
    color: "white",
    fontSize: 32,
    fontWeight: "800",
  },
  tagline: {
    marginTop: 12,
    color: "#e5e7eb",
    fontSize: 14,
  },
});
