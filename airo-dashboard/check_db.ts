import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function check() {
  try {
    const firms = await prisma.firm.findMany()
    const mentions = await prisma.mention.findMany()
    console.log("Firms found:", firms.length)
    console.log("Mentions found:", mentions.length)
    console.log("Sample Firm:", firms[0])
  } catch (e) {
    console.error("Database Error:", e)
  } finally {
    await prisma.$disconnect()
  }
}

check()

