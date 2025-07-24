<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
  codeColClass: {
    type: String,
    default: "overflow-auto",
  },
});
</script>

<template>
  <div class="slidev-layout code-right w-full h-full grid grid-cols-2 gap-8">
    <div class="flex flex-col justify-center p-8" :class="props.class">
      <slot />
    </div>
    <div
      class="flex flex-col justify-center p-4 rounded-lg"
      :class="props.codeColClass"
    >
      <slot name="code" />
    </div>
  </div>
</template>

<style scoped>
.slidev-layout.code-right {
  @apply text-left;
}

.slidev-layout.code-right .left-content {
  @apply flex flex-col justify-center;
}

.slidev-layout.code-right .code-content {
  @apply bg-gray-900 rounded-lg p-4;
}

/* Ensure code blocks fit properly */
.slidev-layout.code-right :deep(pre) {
  @apply m-0 text-sm leading-relaxed;
  max-height: 100%;
  overflow: auto;
}

.slidev-layout.code-right :deep(code) {
  @apply text-sm;
}
</style>
