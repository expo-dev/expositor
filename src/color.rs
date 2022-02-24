pub const W : usize = 0;
pub const B : usize = 1;

pub const WHITE : usize = 0;
pub const BLACK : usize = 8;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
  White = 0,
  Black = 1,
}

impl Color {
  pub fn from_u8(x : u8) -> Self
  {
    debug_assert!(x < 2, "cannot convert u8 to Color");
    return unsafe { std::mem::transmute(x) };
  }

  pub fn from_usize(x : usize) -> Self
  {
    debug_assert!(x < 2, "cannot convert usize to Color");
    return unsafe { std::mem::transmute(x as u8) };
  }

  pub fn as_bool(self) -> bool
  {
    return unsafe { std::mem::transmute(self) };
  }
}

impl std::ops::Not for Color {
  type Output = Self;

  fn not(self) -> Self::Output
  {
    return unsafe { std::mem::transmute(!std::mem::transmute::<_,bool>(self)) };
  }
}

impl std::fmt::Display for Color {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    return f.write_str(["White", "Black"][*self as usize]);
  }
}
