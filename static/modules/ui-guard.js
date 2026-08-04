function asArray(value) {
  if (!value) return [];
  if (typeof value === "string") return [value];
  if (typeof Element !== "undefined" && value instanceof Element) return [value];
  if (typeof value[Symbol.iterator] === "function") return Array.from(value);
  return [value];
}

function asElement(value) {
  if (!value) return null;
  if (typeof value === "string") return document.getElementById(value);
  return value;
}

function restoreAttribute(element, name, value, restore) {
  const hadAttribute = element.hasAttribute(name);
  restore.push(() => {
    if (hadAttribute) {
      element.setAttribute(name, value);
    } else {
      element.removeAttribute(name);
    }
  });
}

export function createAsyncActionGuard() {
  const active = new Set();

  async function run(key, action, options = {}) {
    const guardKey = String(key || "default");
    if (active.has(guardKey)) {
      if (options.duplicateMessage && options.toast) {
        options.toast(options.duplicateMessage, Boolean(options.duplicateError));
      }
      return options.duplicateValue ?? null;
    }

    active.add(guardKey);
    const restore = [];
    const button = asElement(options.button);
    const form = asElement(options.form);
    const controls = new Set();

    if (button) controls.add(button);
    asArray(options.controls).map(asElement).filter(Boolean).forEach((item) => controls.add(item));
    if (form && options.disableForm !== false) {
      form.querySelectorAll("button,input,select,textarea").forEach((item) => controls.add(item));
    }

    [...controls].forEach((element) => {
      if ("disabled" in element) {
        const wasDisabled = element.disabled;
        restore.push(() => { element.disabled = wasDisabled; });
        element.disabled = true;
      }
    });

    [button, form, ...asArray(options.busyElements).map(asElement)].filter(Boolean).forEach((element) => {
      const previous = element.getAttribute("aria-busy");
      restoreAttribute(element, "aria-busy", previous, restore);
      element.setAttribute("aria-busy", "true");
    });

    if (button && options.busyText) {
      const previousText = button.textContent;
      restore.push(() => { button.textContent = previousText; });
      button.textContent = options.busyText;
    }

    if (options.startMessage && options.toast) {
      options.toast(options.startMessage, Boolean(options.startError));
    }

    try {
      return await action();
    } finally {
      for (const restoreOne of restore.reverse()) {
        restoreOne();
      }
      active.delete(guardKey);
    }
  }

  return {
    run,
    isActive: (key) => active.has(String(key || "default")),
  };
}
