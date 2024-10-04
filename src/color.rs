#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
  White = 0,
  Black = 1,
}

use Color::*;
pub const WB : [Color; 2] = [White, Black];

impl std::ops::Not for Color {
  type Output = Self;

  #[inline]
  fn not(self) -> Self::Output
  {
    return unsafe { std::mem::transmute(self as u8 ^ 1) };
  }
}

impl<T> std::ops::Index<Color> for [T] {
  type Output = T;

  #[inline]
  fn index(&self, idx : Color) -> &Self::Output
  {
    return &self[idx as usize];
  }
}

impl<T> std::ops::IndexMut<Color> for [T] {
  #[inline]
  fn index_mut(&mut self, idx : Color) -> &mut Self::Output
  {
    return &mut self[idx as usize];
  }
}

impl std::fmt::Display for Color {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    return f.write_str(["White", "Black"][*self as usize]);
  }
}
