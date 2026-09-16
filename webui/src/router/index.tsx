import { Suspense, lazy } from "react";
import { createBrowserRouter, redirect } from "react-router-dom";
import { consoleRouteObjects } from "./console-routes";
import AlphaLoading from "@/components/alpha-loading";
import { consoleReturnPath, getAuth, loginPath } from "@/api/auth";

const ConsoleLayout = lazy(() => import("@/layouts/console-layout"));
const LoginPage = lazy(() => import("@/views/login"));

export const router = createBrowserRouter([
  {
    path: "/login",
    element: <Suspense fallback={<AlphaLoading fullscreen />}><LoginPage /></Suspense>,
    loader: async ({ request }) => {
      try {
        if ((await getAuth(request.signal)).authenticated)
          return redirect(consoleReturnPath(new URL(request.url).searchParams.get("from")));
      } catch { /* The login page shows request failures and allows retrying. */ }
      return null;
    },
  },
  {
    path: "/",
    loader: async ({ request }) => {
      const url = new URL(request.url);
      try { if ((await getAuth(request.signal)).authenticated) return null; }
      catch { /* Reach the login page when the service is unavailable. */ }
      return redirect(loginPath(url.pathname + url.search + url.hash));
    },
    hydrateFallbackElement: <AlphaLoading fullscreen />,
    element: <Suspense fallback={<AlphaLoading fullscreen />}><ConsoleLayout /></Suspense>,
    children: consoleRouteObjects,
  },
]);
