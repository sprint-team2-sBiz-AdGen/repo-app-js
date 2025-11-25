import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import SplashScreen from "./screens/SplashScreen";
import HomeScreen from "./screens/HomeScreen";
import StrategySelectScreen from "./screens/StrategySelectScreen";
import PhotoAndDescriptionScreen from "./screens/PhotoAndDescriptionScreen";
import GeneratingScreen from "./screens/GeneratingScreen";
import ResultsScreen from "./screens/ResultsScreen";
import ShareScreen from "./screens/ShareScreen";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Splash">
        <Stack.Screen
          name="Splash"
          component={SplashScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: "Feedly AI" }}
        />
        <Stack.Screen
          name="StrategySelect"
          component={StrategySelectScreen}
          options={{ title: "광고 스타일 선택" }}
        />
        <Stack.Screen
          name="PhotoAndDescription"
          component={PhotoAndDescriptionScreen}
          options={{ title: "사진 & 한 줄 설명" }}
        />
        <Stack.Screen
          name="Generating"
          component={GeneratingScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Results"
          component={ResultsScreen}
          options={{ title: "광고 결과" }}
        />
        <Stack.Screen
          name="Share"
          component={ShareScreen}
          options={{ title: "공유하기" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
