"use client";
import { useState, useEffect } from "react";

export default function HomeSection() {
    const [msg, setMsg] = useState("Loading...");
    useEffect(() => {
      fetch("/api")
        .then((res) => res.json())
        .then((data) => {
          setMsg(data.message);
        });
    }, []);
  
    return (
      <main className="w-screen h-screen flex flex-col gap-2 items-center justify-center bg-gradient-to-t from-white to-blue-200">
        <h1 className="font-bold text-2xl">Next JS + Fast API</h1>
        <p>{msg}</p>
      </main>
    );
}