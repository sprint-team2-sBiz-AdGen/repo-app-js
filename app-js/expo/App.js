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

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Splash" // <-- REVERT THIS
        screenOptions={{ headerShown: true }}
      >
        <Stack.Screen name="Splash" component={SplashScreen} />
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="PhotoAndDescription" component={PhotoAndDescriptionScreen} />
        <Stack.Screen name="GeneratingScreen" component={GeneratingScreen} />
        <Stack.Screen name="StrategySelect" component={StrategySelectScreen} />
        <Stack.Screen name="Results" component={ResultsScreen} />
        <Stack.Screen name="Share" component={ShareScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// function App() {
//   return (
//     <NavigationContainer>
//       {/* Temporarily change initialRouteName and add initialParams for testing */}
//       <Stack.Navigator
//         initialRouteName="Results" // <-- CHANGE THIS
//         screenOptions={{ headerShown: true }}
//       >
//         {/* Pass the test jobId as an initial parameter */}
//         <Stack.Screen
//           name="Results"
//           component={ResultsScreen}
//           initialParams={{ jobId: "b02650e3-c70d-40a0-ad86-758aee877682" }} // <-- ADD THIS
//         />

//         {/* Keep other screens for when you revert the changes */}
//         <Stack.Screen name="Splash" component={SplashScreen} />
//         <Stack.Screen name="Home" component={HomeScreen} />
//         <Stack.Screen
//           name="PhotoAndDescription"
//           component={PhotoAndDescriptionScreen}
//         />
//         <Stack.Screen name="GeneratingScreen" component={GeneratingScreen} />
//         <Stack.Screen name="StrategySelect" component={StrategySelectScreen} />
//         {/* <Stack.Screen name="Results" component={ResultsScreen} /> */}
//         <Stack.Screen name="Share" component={ShareScreen} />
//       </Stack.Navigator>
//     </NavigationContainer>
//   );
// }

export default App;