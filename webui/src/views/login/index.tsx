import { useRef, useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { consoleReturnPath, login } from "@/api/auth";
import AlphaLoading from "@/components/alpha-loading";
import { CanvasRevealEffect } from "./login-backdrop";
import "./index.css";

// Form layout and visual treatment adapted from Alpha-Garden-Pro's SignInPage.
export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("123123");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const submitting = useRef(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting.current) return;
    submitting.current = true;
    setPending(true);
    setError("");
    try {
      await login(username, password);
      navigate(consoleReturnPath(new URLSearchParams(location.search).get("from")), {
        replace: true,
      });
    } catch (error) {
      setError(
        error instanceof Error && error.message === "console_credentials_invalid"
          ? "用户名或密码错误，请重新输入。"
          : "暂时无法登录，请确认后端服务已启动后重试。",
      );
    } finally {
      submitting.current = false;
      setPending(false);
    }
  };
  const inputClassName =
    "w-full rounded-full border border-white/10 bg-white/5 px-6 py-4 text-center text-white/70 placeholder:text-white/45 backdrop-blur-[1px] transition-colors focus:border-white/30 focus:outline-none";
  return (
    <main className="ag-login flex w-full min-h-screen flex-col bg-black relative">
      <div className="absolute inset-0 z-0 overflow-hidden" aria-hidden="true">
        <div className="absolute inset-0">
          <CanvasRevealEffect
            animationSpeed={3}
            containerClassName="bg-black"
            colors={[
              [255, 255, 255],
              [255, 255, 255],
            ]}
            dotSize={6}
            reverse={false}
          />
        </div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,0,0,1)_0%,_transparent_100%)]" />
        <div className="absolute top-0 left-0 right-0 h-1/3 bg-gradient-to-b from-black to-transparent" />
      </div>
      <div className="relative z-10 flex min-h-screen items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full max-w-sm space-y-6 text-center"
        >
          <div className="space-y-2">
            <h1 className="text-[2.5rem] font-bold leading-[1.1] tracking-tight text-white">
              Welcome to
              <br />
              Alpha Garden
            </h1>
            <p className="mx-auto max-w-md text-sm leading-6 text-white/55">
              这里可以让 Alpha 变成可培育、可学习、可进化的研究资产
            </p>
          </div>
          <form
            onSubmit={submit}
            className="mx-auto max-w-sm space-y-4"
            aria-label="登录"
            aria-busy={pending}
          >
            <input
              aria-label="用户名"
              type="text"
              name="username"
              placeholder="Username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className={inputClassName}
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              required
              disabled={pending}
            />
            <div className="flex items-center gap-4">
              <div className="h-px bg-white/10 flex-1" />
              <span className="text-white/40 text-sm">and</span>
              <div className="h-px bg-white/10 flex-1" />
            </div>
            <div className="relative">
              <input
                aria-label="密码"
                type="password"
                name="password"
                placeholder="Password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className={`${inputClassName} px-16`}
                autoComplete="current-password"
                required
                disabled={pending}
                aria-describedby={error ? "login-error" : undefined}
              />
              <button
                type="submit"
                aria-label="登录"
                disabled={pending}
                className="absolute right-1.5 top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
              >
                →
              </button>
            </div>
            {error && (
              <p id="login-error" role="alert" className="text-sm text-red-400">
                {error}
              </p>
            )}
          </form>
          <p className="text-xs text-white/40 pt-10">Alpha Garden · 研究控制台</p>
        </motion.div>
      </div>
      {pending && <AlphaLoading fullscreen />}
    </main>
  );
}
